import gym
import torch
import numpy as np
from gymnasium.spaces import Box
from omni.isaac.lab.envs import DirectRLEnv


class MultiEnvWrapper(gym.Wrapper):
    def __init__(self, env: DirectRLEnv):
        super().__init__(env)
        original_action_space = self.env.action_space
        self._action_space = Box(
            low=original_action_space.low[0],
            high=original_action_space.high[0],
            shape=(original_action_space.shape[-1],),
        )
        original_observation_space = self.env.observation_space
        self._observation_space = Box(
            low=original_observation_space.low[0],
            high=original_observation_space.high[0],
            shape=(original_observation_space.shape[-1],),
        )

    def get_obs(self, idx=None, **kwargs):
        if idx is None:
            raw_obs = self.env.unwrapped.get_obs()
            obs = self._process_obs(raw_obs)
            return obs
        else:
            raw_obs = self.env.unwrapped.get_obs()
            obs = self._process_obs(raw_obs)
            return obs[idx]

    def reset(self, **kwargs):
        raw_obs, _ = self.env.reset(**kwargs)
        obs = self._process_obs(raw_obs)
        return obs

    def step(self, action):
        """
        return
            next observation,
            reward,
            done,
            if episode has done without timeout,
            extra information (such as success)
        """
        if isinstance(action, np.ndarray):
            action = torch.FloatTensor(action).to(self.device)

        obs_raw, rew, terminated, timeout, raw_extra = self.env.step(action)

        obs = self._process_obs(obs_raw)
        done = terminated | timeout
        done_no_max = terminated & ~timeout
        extra = self._process_extra(raw_extra)

        return obs, rew, done, done_no_max, extra

    def _process_obs(self, raw_obs):
        obs = raw_obs["policy"]
        return obs

    def _process_extra(self, raw_extra):
        extra = dict()
        for key, value in raw_extra:
            extra[key] = value
        return extra
