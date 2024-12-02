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

import numpy as np
import torch
import os
import time
import gymnasium as gym
import hydra

import envs
import my_utils

from my_utils import Logger
from my_utils import ReplayBuffer
from my_utils import MultiEnvWrapper
from reward_model.reward_model_scriptedT import RewardModel
from collections import deque


from torch.profiler import profile, ProfilerActivity


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

        env = gym.make(
            cfg.env,
            seed=cfg.seed,
            device=cfg.device,
            num_envs=cfg.num_envs,
            render_mode="rgb_array" if args_cli.video else None,
        )
        if env.unwrapped.viewport_camera_controller != None:
            env.unwrapped.viewport_camera_controller.update_view_location(
                [-6, -3, 3], [2, 0, 2]
            )
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

        self.replay_buffer = ReplayBuffer(
            obs_shape=self.env.observation_space.shape,
            action_shape=self.env.action_space.shape,
            capacity=int(cfg.replay_buffer_capacity),
            device=self.device,
            window=self.num_envs,
        )

        # for logging
        self.total_feedback = 0
        self.labeled_feedback = 0
        self.step = 0

        # instantiating the reward model
        self.reward_model = RewardModel(
            self.env.observation_space.shape[0],
            self.env.action_space.shape[0],
            ensemble_size=cfg.ensemble_size,
            size_segment=cfg.segment,
            activation=cfg.activation,
            lr=cfg.reward_lr,
            mb_size=cfg.reward_batch,
            large_batch=cfg.large_batch,
            label_margin=cfg.label_margin,
            teacher_beta=cfg.teacher_beta,
            teacher_gamma=cfg.teacher_gamma,
            teacher_eps_mistake=cfg.teacher_eps_mistake,
            teacher_eps_skip=cfg.teacher_eps_skip,
            teacher_eps_equal=cfg.teacher_eps_equal,
        )

    def learn_reward(self, first_flag=0):

        # get feedbacks
        labeled_queries, noisy_queries = 0, 0
        if first_flag == 1:
            # if it is first time to get feedback, need to use random sampling
            labeled_queries = self.reward_model.uniform_sampling()
        else:
            if self.cfg.feed_type == 0:
                labeled_queries = self.reward_model.uniform_sampling()
            elif self.cfg.feed_type == 1:
                labeled_queries = self.reward_model.disagreement_sampling()
            elif self.cfg.feed_type == 2:
                labeled_queries = self.reward_model.entropy_sampling()
            elif self.cfg.feed_type == 3:
                labeled_queries = self.reward_model.kcenter_sampling()
            elif self.cfg.feed_type == 4:
                labeled_queries = self.reward_model.kcenter_disagree_sampling()
            elif self.cfg.feed_type == 5:
                labeled_queries = self.reward_model.kcenter_entropy_sampling()
            else:
                raise NotImplementedError

        self.total_feedback += self.reward_model.mb_size
        self.labeled_feedback += labeled_queries

        train_acc = 0
        if self.labeled_feedback > 0:
            # update reward
            for _ in range(self.cfg.reward_update):
                if self.cfg.label_margin > 0 or self.cfg.teacher_eps_equal > 0:
                    train_acc = self.reward_model.train_soft_reward()
                else:
                    train_acc = self.reward_model.train_reward()
                total_acc = np.mean(train_acc)

                if total_acc > 0.97:
                    break

        print("Reward function is updated!! ACC: " + str(total_acc))

    def run(self):
        episode = 0
        model_episode_reward = np.zeros(self.num_envs)
        true_episode_reward = np.zeros(self.num_envs)
        episode_done = np.zeros(self.num_envs)

        obs = self.env.reset()
        obs_np = obs.detach().cpu().numpy()

        avg_train_true_return = deque([], maxlen=10)

        obs_query = [[] for _ in range(self.num_envs)]
        action_query = [[] for _ in range(self.num_envs)]
        reward_query = [[] for _ in range(self.num_envs)]

        interact_count = 0
        while self.step < self.cfg.num_train_steps:
            # reset done environment
            done_idx = np.where(episode_done)[0]
            if done_idx.size != 0:

                self.logger.log(
                    "train/episode_reward",
                    true_episode_reward[done_idx].sum().item() / done_idx.size,
                    self.step,
                )
                self.logger.log(
                    "train/model_episode_reward",
                    true_episode_reward[done_idx].sum().item() / done_idx.size,
                    self.step,
                )
                self.logger.log("train/total_feedback", self.total_feedback, self.step)

                for i in done_idx:
                    self.reward_model.add_data(
                        np.array(obs_query[i]),
                        np.array(action_query[i]),
                        np.array(reward_query[i]).reshape(-1, 1),
                    )
                    obs_query[i] = []
                    action_query[i] = []
                    reward_query[i] = []

                obs[done_idx] = self.env.reset(done_idx)
                obs_np = obs.detach().cpu().numpy()
                model_episode_reward[done_idx] = 0
                avg_train_true_return.extend(true_episode_reward[done_idx])
                true_episode_reward[done_idx] = 0
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
            if self.step == (self.cfg.num_seed_steps + self.cfg.num_unsup_steps):
                # update schedule
                if self.cfg.reward_schedule == 1:
                    frac = (
                        self.cfg.num_train_steps - self.step
                    ) / self.cfg.num_train_steps
                    if frac == 0:
                        frac = 0.01
                elif self.cfg.reward_schedule == 2:
                    frac = self.cfg.num_train_steps / (
                        self.cfg.num_train_steps - self.step + 1
                    )
                else:
                    frac = 1
                self.reward_model.change_batch(frac)

                # first learn reward
                self.learn_reward(first_flag=1)

                # relabel buffer
                self.replay_buffer.relabel_with_predictor(self.reward_model)

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

                # reset interact_count
                interact_count = 0

            elif self.step > self.cfg.num_seed_steps + self.cfg.num_unsup_steps:
                # update reward function
                if self.total_feedback < self.cfg.max_feedback:
                    if interact_count >= self.cfg.num_interact:
                        # update schedule
                        if self.cfg.reward_schedule == 1:
                            frac = (
                                self.cfg.num_train_steps - self.step
                            ) / self.cfg.num_train_steps
                            if frac == 0:
                                frac = 0.01
                        elif self.cfg.reward_schedule == 2:
                            frac = self.cfg.num_train_steps / (
                                self.cfg.num_train_steps - self.step + 1
                            )
                        else:
                            frac = 1
                        self.reward_model.change_batch(frac)

                        # corner case: new total feed > max feed
                        if (
                            self.reward_model.mb_size + self.total_feedback
                            > self.cfg.max_feedback
                        ):
                            self.reward_model.set_batch(
                                self.cfg.max_feedback - self.total_feedback
                            )

                        self.learn_reward()
                        self.replay_buffer.relabel_with_predictor(self.reward_model)
                        interact_count = 0

                self.agent.update(self.replay_buffer, self.logger, self.step, 1)

            # unsupervised exploration
            elif self.step > self.cfg.num_seed_steps:
                self.agent.update_state_ent(
                    self.replay_buffer,
                    self.logger,
                    self.step,
                    gradient_update=1,
                    K=self.cfg.topK,
                )

            next_obs, reward, done, done_no_max, _ = self.env.step(action)
            reward_hat = self.reward_model.r_hat_tensor(
                torch.cat([obs, action], axis=-1)
            ).squeeze(-1)

            # adding data to the reward training data
            next_obs_np = next_obs.detach().cpu().numpy()
            reward_np = reward.detach().cpu().numpy()
            reward_hat_np = reward_hat.detach().cpu().numpy()
            done_np = done.detach().cpu().numpy()
            done_no_max_np = done_no_max.detach().cpu().numpy()

            self.replay_buffer.add_batch(
                obs_np,
                action_np,
                reward_np.reshape(-1, 1),
                next_obs_np,
                done_np.reshape(-1, 1),
                done_no_max_np.reshape(-1, 1),
            )

            episode_done = done_np
            model_episode_reward += reward_hat_np
            true_episode_reward += reward_np

            # update obs_query, action_query to be used to add in reward_model
            for i in range(self.num_envs):
                obs_query[i].append(obs_np[i])
                action_query[i].append(action_np[i])
                reward_query[i].append(reward_np[i])

            obs = next_obs
            obs_np = next_obs_np
            self.step += self.num_envs
            interact_count += self.num_envs

        self.agent.save(self.work_dir, self.step)
        self.reward_model.save(self.work_dir, self.step)


import cProfile
import pstats


@hydra.main(
    config_path="config", config_name="train_PEBBLE_scriptedT", version_base="1.1"
)
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


import cProfile

if __name__ == "__main__":
    main()
    simulation_app.close()
