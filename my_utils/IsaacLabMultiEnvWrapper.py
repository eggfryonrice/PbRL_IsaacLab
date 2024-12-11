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
            obs, body_state = self._process_policy_obs(raw_obs)
            return obs, body_state
        else:
            raw_obs = self.env.unwrapped.get_obs()
            obs, body_state = self._process_policy_obs(raw_obs)
            return obs[idx], body_state[idx]

    def reset(self, **kwargs):
        raw_obs, _ = self.env.reset(**kwargs)
        obs, body_state = self._process_policy_obs(raw_obs)
        return obs, body_state

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

        obs_raw, rew, terminated, timeout, _ = self.env.step(action)

        obs, body_state = self._process_policy_obs(obs_raw)
        done = terminated | timeout
        done_no_max = terminated & ~timeout

        return obs, rew, done, done_no_max, body_state

    def _process_policy_obs(self, raw_obs):
        return raw_obs["policy"], raw_obs["body_state"]
