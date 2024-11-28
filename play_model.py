import argparse
import sys
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="isaac sim app related parser")
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos during training."
)
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

from IsaacLabMultiEnvWrapper import MultiEnvWrapper

import envs
import my_utils


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f"workspace: {self.work_dir}")

        self.cfg = cfg

        my_utils.set_seed_everywhere(cfg.seed)

        self.device = torch.device(cfg.device)

        self.num_envs = 1
        env = gym.make(
            cfg.env,
            seed=cfg.seed,
            device=cfg.device,
            num_envs=self.num_envs,
            render_mode="rgb_array",
        )
        if env.viewport_camera_controller != None:
            env.viewport_camera_controller.update_view_location([-6, -6, 3], [2, 0, 2])
        if args_cli.video:
            video_kwargs = {
                "video_folder": os.path.join(self.work_dir, "videos"),
                "step_trigger": lambda step: step % 500 == 0,
                "video_length": 500,
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
        self.agent.load(self.work_dir, int(self.cfg.num_train_steps))
        # self.agent.load(self.work_dir, 10)

    def run(self):
        episode_done = np.zeros(self.num_envs)
        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        steps = 0

        while steps < 1000:
            # reset done environment
            done_idx = np.where(episode_done)[0]
            if done_idx.size != 0:
                obs[done_idx], _ = self.env.reset(done_idx)

            with my_utils.eval_mode(self.agent):
                action = self.agent.act(obs, sample=False)

            step_result = self.env.step(action)

            next_obs, done = step_result[0], step_result[2]

            done_np = done.float().detach().cpu().numpy()

            episode_done = done_np

            obs = next_obs

            steps += self.num_envs


@hydra.main(config_path="config", config_name="finetune_PEBBLE", version_base="1.1")
def main(cfg):

    workspace = Workspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
    simulation_app.close()
