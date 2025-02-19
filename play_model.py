import argparse
import sys
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="isaac sim app related parser")
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos during training."
)
parser.add_argument(
    "--config_name",
    type=str,
    default="train_PEBBLE_scriptedT",
    help="Hydra config name.",
)
parser.add_argument(
    "--step",
    type=int,
    default=4500000,
    help="model at which train step.",
)
parser.add_argument(
    "--mirror", action="store_true", default=False, help="Enable mirroring."
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

import envs
import my_utils

from my_utils import MultiEnvWrapper


class Workspace(object):
    def __init__(self, cfg, mirror, step):
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
        if env.unwrapped.viewport_camera_controller != None:
            env.unwrapped.viewport_camera_controller.update_view_location(
                [8, 0, 2],
                [2, 0, 1],
                # [2, -6, 2],
                # [2, 0, 1],
            )
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
        # self.agent.load(self.work_dir, int(self.cfg.num_train_steps))
        self.agent.load(self.work_dir, step)

        self.mirror = mirror

    def run(self):
        episode_done = np.zeros(self.num_envs)
        obs, _ = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        steps = 0

        while steps < 10000:
            # reset done environment
            done_idx = np.where(episode_done)[0]
            if done_idx.size != 0:
                obs[done_idx], _ = self.env.get_obs(done_idx)

            if self.mirror:
                obs[0] = torch.tensor(
                    self.env.get_mirrored_state(np.array(obs[0].cpu())),
                    device=self.device,
                )

            with my_utils.eval_mode(self.agent):
                action = self.agent.act(obs, sample=False)

            if self.mirror:
                action[0] = torch.tensor(
                    self.env.get_mirrored_action(np.array(action[0].detach().cpu())),
                    device=self.device,
                )

            # action = torch.zeros_like(action)
            # action[0][20] = 1

            step_result = self.env.step(action)

            next_obs, done = step_result[0], step_result[2]

            done_np = done.float().detach().cpu().numpy()

            episode_done = done_np

            obs = next_obs

            steps += self.num_envs


@hydra.main(config_path="config", config_name=args_cli.config_name, version_base="1.1")
def main(cfg):

    workspace = Workspace(cfg, args_cli.mirror, args_cli.step)
    workspace.run()


if __name__ == "__main__":
    main()
    simulation_app.close()
