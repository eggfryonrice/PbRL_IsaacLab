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
import os
import time
import pickle as pkl
import my_utils
import hydra
import gymnasium as gym

from logger import Logger
from replay_buffer import ReplayBuffer
from IsaacLabSingleEnvWrapper import SimpleEnvWrapper

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
        self.log_success = False
        self.step = 0

        env = gym.make(
            cfg.env,
            seed=cfg.seed,
            device=cfg.device,
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
        self.env = SimpleEnvWrapper(env)

        cfg.agent.obs_dim = self.env.observation_space.shape[0]
        cfg.agent.action_dim = self.env.action_space.shape[0]
        cfg.agent.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max()),
        ]
        self.agent = hydra.utils.instantiate(cfg.agent)

        # no relabel
        self.replay_buffer = ReplayBuffer(
            self.env.observation_space.shape,
            self.env.action_space.shape,
            int(cfg.replay_buffer_capacity),
            self.device,
        )
        meta_file = os.path.join(self.work_dir, "metadata.pkl")
        pkl.dump({"cfg": self.cfg}, open(meta_file, "wb"))

    def evaluate(self):
        average_episode_reward = 0
        if self.log_success:
            success_rate = 0

        for _ in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()
            self.agent.reset()
            done = False
            episode_reward = 0

            if self.log_success:
                episode_success = 0

            while not done:
                with my_utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, _, extra = self.env.step(action)
                episode_reward += reward
                if self.log_success:
                    episode_success = max(episode_success, extra["success"])

            average_episode_reward += episode_reward
            if self.log_success:
                success_rate += episode_success

        average_episode_reward /= self.cfg.num_eval_episodes
        if self.log_success:
            success_rate /= self.cfg.num_eval_episodes
            success_rate *= 100.0

        self.logger.log("eval/episode_reward", average_episode_reward, self.step)

        if self.log_success:
            self.logger.log("eval/success_rate", success_rate, self.step)
        self.logger.dump(self.step)

    def run(self):
        episode, episode_reward, done = 0, 0, True
        if self.log_success:
            episode_success = 0
        start_time = time.time()
        eval_count = 1

        while self.step < self.cfg.num_train_steps:
            if done:
                if self.step > 0:
                    self.logger.log(
                        "train/duration", time.time() - start_time, self.step
                    )
                    start_time = time.time()
                    self.logger.dump(
                        self.step, save=(self.step > self.cfg.num_seed_steps)
                    )

                # evaluate agent periodically
                if (
                    self.step - (self.cfg.num_seed_steps + self.cfg.num_unsup_steps)
                    >= eval_count * self.cfg.eval_frequency
                ):
                    self.logger.log("eval/episode", episode, self.step)
                    self.evaluate()
                    eval_count += 1

                self.logger.log("train/episode_reward", episode_reward, self.step)

                if self.log_success:
                    self.logger.log("train/episode_success", episode_success, self.step)

                obs = self.env.reset()
                self.agent.reset()
                done = False
                episode_reward = 0
                if self.log_success:
                    episode_success = 0
                episode += 1

                self.logger.log("train/episode", episode, self.step)

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with my_utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

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
                self.agent.update(self.replay_buffer, self.logger, self.step)
            # unsupervised exploration
            elif self.step > self.cfg.num_seed_steps:
                self.agent.update_state_ent(
                    self.replay_buffer,
                    self.logger,
                    self.step,
                    gradient_update=1,
                    K=self.cfg.topK,
                )

            next_obs, reward, done, done_no_max, extra = self.env.step(action)
            # allow infinite bootstrap
            done = float(done)
            done_no_max = float(done_no_max)
            episode_reward += reward

            if self.log_success:
                episode_success = max(episode_success, extra["success"])

            self.replay_buffer.add(obs, action, reward, next_obs, done, done_no_max)

            obs = next_obs
            self.step += 1

        self.agent.save(self.work_dir, self.step)


@hydra.main(config_path="config", config_name="train_SAC", version_base="1.1")
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
    simulation_app.close()
