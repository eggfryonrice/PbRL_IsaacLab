import argparse
import sys
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="isaac sim app related parser")
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos during training."
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import numpy as np
import os
import time
import pickle as pkl
import my_utils
import hydra
import gymnasium as gym

from logger import Logger
from replay_buffer import ReplayBuffer
from IsaacLabMultiEnvWrapper import MultiEnvWrapper
import envs


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f"workspace: {self.work_dir}")

        self.cfg = cfg
        self.logger = Logger(
            self.work_dir,
            save_tb=cfg.log_save_tb,
            log_frequency=cfg.log_frequency,
            agent=cfg.agent_name,
        )

        my_utils.set_seed_everywhere(cfg.seed)

        self.device = torch.device(cfg.device)
        self.num_envs = cfg.num_envs
        self.step = 0

        env = gym.make(
            cfg.env,
            seed=cfg.seed,
            device=cfg.device,
            num_envs=cfg.num_envs,
            render_mode="rgb_array" if args_cli.video else None,
        )
        if env.viewport_camera_controller != None:
            env.viewport_camera_controller.update_view_location([-6, -3, 3], [2, 0, 2])
        if args_cli.video:
            video_kwargs = {
                "video_folder": os.path.join(self.work_dir, "videos", "train"),
                "step_trigger": lambda step: step % self.cfg.video_interval == 0,
                "video_length": self.cfg.video_length,
                "disable_logger": True,
            }
            print("[INFO]: Recording videos during training.")
            env = gym.wrappers.RecordVideo(env, **video_kwargs)
        self.env = MultiEnvWrapper(env)

        cfg.agent.obs_dim = self.env.observation_space.shape[0]
        cfg.agent.action_dim = self.env.action_space.shape[0]
        cfg.agent.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max()),
        ]
        self.agent = hydra.utils.instantiate(cfg.agent)

        # no relabel
        self.replay_buffer = ReplayBuffer(
            obs_shape=self.env.observation_space.shape,
            action_shape=self.env.action_space.shape,
            capacity=int(cfg.replay_buffer_capacity),
            device=self.device,
            window=self.num_envs,
        )
        meta_file = os.path.join(self.work_dir, "metadata.pkl")
        pkl.dump({"cfg": self.cfg}, open(meta_file, "wb"))

    def run(self):
        episode = 0
        episode_reward = np.zeros(self.num_envs)
        episode_done = np.zeros(self.num_envs)

        obs = self.env.reset()
        obs_np = obs.detach().cpu().numpy()

        while self.step < self.cfg.num_train_steps:
            # reset done environment
            done_idx = np.where(episode_done)[0]
            if done_idx.size != 0:
                obs[done_idx] = self.env.reset(done_idx)
                obs_np = obs.detach().cpu().numpy()
                self.logger.log(
                    "train/episode_reward",
                    episode_reward[done_idx].sum() / done_idx.size,
                    self.step,
                )
                episode_reward[done_idx] = 0
                episode += done_idx.size
                self.logger.log("train/episode", episode, self.step)
                self.logger.dump(self.step, save=(self.step > self.cfg.num_seed_steps))

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = torch.tensor(
                    np.array(
                        [self.env.action_space.sample() for _ in range(self.num_envs)]
                    ),
                    dtype=torch.float32,
                    device=self.device,
                )
            else:
                with my_utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)
            action_np = action.detach().cpu().numpy()

            # run training update
            if (
                self.step == (self.cfg.num_seed_steps + self.cfg.num_unsup_steps)
                and self.cfg.num_unsup_steps > 0
            ):
                # reset Q due to unsuperivsed exploration
                self.agent.reset_critic(self.cfg.double_q_critic)
                # update agent
                self.agent.update_after_reset(
                    self.replay_buffer,
                    self.logger,
                    self.step,
                    gradient_update=self.cfg.reset_update,
                    policy_update=True,
                )
            elif self.step > self.cfg.num_seed_steps + self.cfg.num_unsup_steps:
                self.agent.update(
                    replay_buffer=self.replay_buffer,
                    logger=self.logger,
                    step=self.step,
                    gradient_update=self.cfg.num_gradient_update,
                )
            # unsupervised exploration
            elif self.step > self.cfg.num_seed_steps:
                self.agent.update_state_ent(
                    self.replay_buffer,
                    self.logger,
                    self.step,
                    gradient_update=self.cfg.num_gradient_update,
                    K=self.cfg.topK,
                )

            next_obs, reward, done, done_no_max, _ = self.env.step(action)

            next_obs_np = next_obs.detach().cpu().numpy()
            reward_np = reward.detach().cpu().numpy()
            done_np = done.float().detach().cpu().numpy()
            done_no_max_np = done_no_max.float().detach().cpu().numpy()

            self.replay_buffer.add_batch(
                obs_np,
                action_np,
                reward_np.reshape(-1, 1),
                next_obs_np,
                done_np.reshape(-1, 1),
                done_no_max_np.reshape(-1, 1),
            )

            episode_done = done_np
            episode_reward += reward_np

            obs = next_obs
            obs_np = next_obs_np
            self.step += self.num_envs

        self.agent.save(self.work_dir, self.step)


import cProfile
import pstats


@hydra.main(config_path="config", config_name="train_SAC", version_base="1.1")
def main(cfg):
    profiler = cProfile.Profile()
    profiler.enable()

    workspace = Workspace(cfg)
    workspace.run()

    profiler.disable()

    # Save to a file
    with open("profile_results.txt", "w") as f:
        stats = pstats.Stats(profiler, stream=f)
        stats.strip_dirs()
        stats.sort_stats("cumtime")
        stats.print_stats()


if __name__ == "__main__":
    main()
    simulation_app.close()
