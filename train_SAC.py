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
import time
import hydra

from BaseWorkspace import BaseWorkspace


class Workspace(BaseWorkspace):
    def __init__(self, cfg):
        super().__init__(cfg, args_cli.video)

        self.initialize_replay_buffer()

    def run(self):
        self.initialize_running()

        while self.step < self.cfg.num_train_steps:
            # reset done environment
            done_idx = np.where(self.episode_done)[0]
            if done_idx.size != 0:
                self.handle_done(done_idx)
                self.logger.dump(self.step, save=(self.step > self.cfg.num_seed_steps))

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                self.set_random_action()
            else:
                self.set_agent_action()

            # run training update
            if (
                self.step == (self.cfg.num_seed_steps + self.cfg.num_unsup_steps)
                and self.cfg.num_unsup_steps > 0
            ):
                self.update_agent_after_unsupervised_pretraining()
            elif self.step > self.cfg.num_seed_steps + self.cfg.num_unsup_steps:
                self.update_agent()
            elif self.step > self.cfg.num_seed_steps:
                self.update_agent_during_unsupervised_pretraining()

            self.environment_step()

            self.replay_buffer.add_batch(
                self.obs_np,
                self.action_np,
                self.reward_np.reshape(-1, 1),
                self.next_obs_np,
                self.done_np.reshape(-1, 1),
                self.done_no_max_np.reshape(-1, 1),
            )

            self.reallocate_datas()

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
