import argparse
import sys
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="isaac sim app related parser")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
import time
import hydra


from my_utils import ReplayBuffer
from base_workspace import BaseWorkspace


class Workspace(BaseWorkspace):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.initialize_combined_replay_buffer()

        self.total_feedback = 0
        self.labeled_feedback = 0
        self.step = 0

        # instantiating the reward model
        self.initialize_reward_model_humanT()

    def run(self):
        self.initialize_running()
        self.model_episode_reward = np.zeros(self.num_envs)

        self.obs_query = [[] for _ in range(self.num_envs)]
        self.body_state_query = [[] for _ in range(self.num_envs)]
        self.action_query = [[] for _ in range(self.num_envs)]

        self.interact_count = 0

        while self.step < self.cfg.num_train_steps:
            # reset done environment
            done_idx = np.where(self.episode_done)[0]
            if done_idx.size != 0:
                self.handle_done(done_idx)
                self.logger.log(
                    "train/model_episode_reward",
                    self.model_episode_reward[done_idx].sum().item() / done_idx.size,
                    self.step,
                )
                self.logger.log("train/total_feedback", self.total_feedback, self.step)

                self.model_episode_reward[done_idx] = 0

                if self.total_feedback < self.cfg.max_feedback:
                    for i in done_idx:
                        self.reward_model.add_data(
                            np.array(self.obs_query[i]),
                            np.array(self.action_query[i]),
                            np.array(self.body_state_query[i]),
                        )
                        self.obs_query[i] = []
                        self.body_state_query[i] = []
                        self.action_query[i] = []

                self.logger.log("train/episode", self.episode, self.step)
                self.logger.dump(self.step, save=(self.step > self.cfg.num_seed_steps))

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                self.set_random_action()
            else:
                self.set_agent_action()

            # update obs_query, action_query to be used to add in reward_model
            for i in range(self.num_envs):
                self.obs_query[i].append(self.obs_np[i])
                self.body_state_query[i].append(self.body_state_np[i])
                self.action_query[i].append(self.action_np[i])

            # run training update
            if self.step == (self.cfg.num_seed_steps + self.cfg.num_unsup_steps):
                # first learn reward
                self.learn_reward(first_flag=1)

                # relabel buffer
                self.replay_buffer.relabel_combined_with_predictor(self.reward_model)

                # reset Q due to unsuperivsed exploration
                self.update_agent_after_unsupervised_pretraining()

                # reset interact_count
                self.interact_count = 0

            elif self.step > self.cfg.num_seed_steps + self.cfg.num_unsup_steps:
                # update reward function
                if self.total_feedback < self.cfg.max_feedback:
                    if self.interact_count >= self.cfg.num_interact:
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
                        self.interact_count = 0

                self.update_agent()

            # unsupervised exploration
            elif self.step > self.cfg.num_seed_steps:
                self.update_agent_during_unsupervised_pretraining()

            self.environment_step()

            self.reward_hat = self.reward_model.r_hat_tensor(
                torch.cat([self.obs, self.action], axis=-1)
            ).squeeze(-1)
            self.reward_hat_np = self.reward_hat.detach().cpu().numpy()

            self.replay_buffer.add_combined_batch(
                self.obs_np,
                self.action_np,
                self.reward_np.reshape(-1, 1),
                self.reward_hat_np.reshape(-1, 1),
                self.next_obs_np,
                self.done_np.reshape(-1, 1),
                self.done_no_max_np.reshape(-1, 1),
            )
            if self.cfg.mirror:
                mirrored_obs = np.zeros_like(self.obs_np)
                mirrored_next_obs = np.zeros_like(self.next_obs_np)
                mirrored_action = np.zeros_like(self.action_np)

                for i in range(self.num_envs):
                    mirrored_obs[i] = self.env.unwrapped.get_mirrored_state(
                        self.obs_np[i]
                    )
                    mirrored_next_obs[i] = self.env.unwrapped.get_mirrored_state(
                        self.next_obs_np[i]
                    )
                    mirrored_action[i] = self.env.unwrapped.get_mirrored_action(
                        self.action_np[i]
                    )
                self.replay_buffer.add_combined_batch(
                    mirrored_obs,
                    mirrored_action,
                    self.reward_np.reshape(-1, 1),
                    self.reward_hat_np.reshape(-1, 1),
                    mirrored_next_obs,
                    self.done_np.reshape(-1, 1),
                    self.done_no_max_np.reshape(-1, 1),
                )

            self.reallocate_datas()
            self.model_episode_reward += self.reward_hat_np

            self.step += self.num_envs
            self.interact_count += self.num_envs

            if self.step % self.cfg.save_model_freq == 0:
                self.agent.save(self.work_dir, self.step)
                self.reward_model.save(self.work_dir, self.step)

        self.agent.save(self.work_dir, self.step)
        self.reward_model.save(self.work_dir, self.step)


import cProfile
import pstats


@hydra.main(config_path="config", config_name="train_PEBBLE_humanT", version_base="1.1")
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
