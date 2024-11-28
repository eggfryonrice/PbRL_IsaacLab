import argparse
import sys
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="isaac sim app related parser")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
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

from logger import Logger
from replay_buffer import ReplayBuffer
from reward_model_humanT import RewardModel
from collections import deque
from IsaacLabMultiEnvWrapper import MultiEnvWrapper

import envs
import my_utils


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f"workspace: {self.work_dir}")
        self.video_path = os.path.join(self.work_dir, "video.h5")
        my_utils.initialize_h5_file(self.video_path)

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
            render_mode="rgb_array",
        )
        self.env = MultiEnvWrapper(env)

        cfg.agent.obs_dim = self.env.observation_space.shape[0]
        cfg.agent.action_dim = self.env.action_space.shape[0]
        cfg.agent.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max()),
        ]
        self.agent = hydra.utils.instantiate(cfg.agent)
        self.agent.load(cfg.pretrained_model_dir, cfg.pretrained_model_step)

        self.replay_buffer = ReplayBuffer(
            obs_shape=self.env.observation_space.shape,
            action_shape=self.env.action_space.shape,
            capacity=int(cfg.replay_buffer_capacity),
            device=self.device,
            window=self.num_envs,
            alpha=cfg.reward_alpha,
            beta=cfg.reward_beta,
        )

        # for logging
        self.total_feedback = 0
        self.labeled_feedback = 0
        self.step = 0

        # instantiating the reward model
        self.reward_model = RewardModel(
            video_path=self.video_path,
            dt=self.env.unwrapped.step_dt,
            ds=self.env.observation_space.shape[0],
            da=self.env.action_space.shape[0],
            ensemble_size=cfg.ensemble_size,
            lr=cfg.reward_lr,
            mb_size=cfg.reward_batch,
            size_segment=cfg.segment,
            activation=cfg.activation,
            large_batch=cfg.large_batch,
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

        obs, pic = self.env.reset()
        obs_np = obs.detach().cpu().numpy()
        pic_np = pic.detach().cpu().numpy()

        pic_query = [[] for _ in range(self.num_envs)]
        obs_query = [[] for _ in range(self.num_envs)]
        action_query = [[] for _ in range(self.num_envs)]

        avg_train_true_return = deque([], maxlen=10)

        frame_save_cnt = 0
        interact_count = 0

        while self.step < self.cfg.num_train_steps:
            # reset done environment
            done_idx = np.where(episode_done)[0]
            if done_idx.size != 0:
                # print(
                #     f"Step {self.step}: Allocated: {torch.cuda.memory_allocated() / 1e6} MB, Reserved: {torch.cuda.memory_reserved() / 1e6} MB"
                # )
                self.logger.log(
                    "train/episode_reward",
                    true_episode_reward[done_idx].sum().item() / done_idx.size,
                    self.step,
                )
                self.logger.log(
                    "train/model_episode_reward",
                    model_episode_reward[done_idx].sum().item() / done_idx.size,
                    self.step,
                )
                self.logger.log("train/total_feedback", self.total_feedback, self.step)

                if self.total_feedback < self.cfg.max_feedback:
                    for i in done_idx:
                        my_utils.save_frames(
                            self.video_path, frame_save_cnt, pic_query[i]
                        )
                        frame_save_cnt += 1
                        pic_query[i] = []

                        self.reward_model.add_data(
                            np.array(obs_query[i]), np.array(action_query[i])
                        )
                        obs_query[i] = []
                        action_query[i] = []

                obs[done_idx], pic[done_idx] = self.env.reset(done_idx)
                obs_np = obs.detach().cpu().numpy()
                pic_np = pic.detach().cpu().numpy()
                model_episode_reward[done_idx] = 0
                avg_train_true_return.extend(true_episode_reward[done_idx])
                true_episode_reward[done_idx] = 0
                episode += done_idx.size

                self.logger.log("train/episode", episode, self.step)
                self.logger.dump(self.step, save=(self.step > self.cfg.num_seed_steps))

            # sample action for data collection
            action = self.agent.act(obs, sample=True)
            action_np = action.detach().cpu().numpy()

            # update obs_query, action_query and pic_query to be used to add in reward_model
            for i in range(self.num_envs):
                obs_query[i].append(obs_np[i])
                action_query[i].append(action_np[i])
                pic_query[i].append(pic_np[i])

            # run training update
            if self.step == self.cfg.num_seed_steps:
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

                # reset interact count
                interact_count = 0

            elif self.step > self.cfg.num_seed_steps:
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
                        self.replay_buffer.relabel_combined_with_predictor(
                            self.reward_model
                        )
                        interact_count = 0

                self.agent.update(self.replay_buffer, self.logger, self.step, 1)

            next_obs, reward, done, done_no_max, _, pic = self.env.step(action)
            reward_hat = self.reward_model.r_hat_tensor(
                torch.cat([obs, action], axis=-1)
            ).squeeze(-1)

            # adding data to the reward training data
            next_obs_np = next_obs.detach().cpu().numpy()
            reward_np = reward.detach().cpu().numpy()
            reward_hat_np = reward_hat.detach().cpu().numpy()
            done_np = done.float().detach().cpu().numpy()
            done_no_max_np = done_no_max.float().detach().cpu().numpy()
            pic_np = pic.detach().cpu().numpy()

            self.replay_buffer.add_combined_batch(
                obs_np,
                action_np,
                reward_np.reshape(-1, 1),
                reward_hat_np.reshape(-1, 1),
                next_obs_np,
                done_np.reshape(-1, 1),
                done_no_max_np.reshape(-1, 1),
            )

            episode_done = done_np
            model_episode_reward += reward_hat_np
            true_episode_reward += reward_np

            obs = next_obs
            obs_np = next_obs_np
            self.step += self.num_envs
            interact_count += self.num_envs

            if self.step % 100000 == 0:
                self.agent.save(self.work_dir, self.step)

        self.reward_model.save(self.work_dir, self.step)


import cProfile
import pstats


@hydra.main(config_path="config", config_name="finetune_PEBBLE", version_base="1.1")
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