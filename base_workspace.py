import torch
import numpy as np
import os
import time
import pickle as pkl
import gymnasium as gym
import envs
import my_utils
import hydra

from my_utils import Logger
from my_utils import ReplayBuffer
from my_utils import MultiEnvWrapper
import reward_model.reward_model_humanT
import reward_model.reward_model_scriptedT


class BaseWorkspace(object):
    def __init__(self, cfg, video=False):
        self.work_dir = os.getcwd()
        print(f"workspace: {self.work_dir}")

        self.cfg = cfg
        self.initialize_logger()

        my_utils.set_seed_everywhere(cfg.seed)

        self.device = torch.device(cfg.device)
        self.num_envs = cfg.num_envs
        self.step = 0

        self.initialize_env(video)

        self.initialize_agent()

        meta_file = os.path.join(self.work_dir, "metadata.pkl")
        pkl.dump({"cfg": self.cfg}, open(meta_file, "wb"))

    def initialize_logger(self):
        self.logger = Logger(
            self.work_dir,
            save_tb=self.cfg.log_save_tb,
            log_frequency=self.cfg.log_frequency,
            agent=self.cfg.agent_name,
        )

    def initialize_env(self, video):
        env = gym.make(
            self.cfg.env,
            seed=self.cfg.seed,
            device=self.cfg.device,
            num_envs=self.cfg.num_envs,
            render_mode="rgb_array" if video else None,
        )
        if env.unwrapped.viewport_camera_controller != None:
            env.unwrapped.viewport_camera_controller.update_view_location(
                [-6, -3, 3], [2, 0, 2]
            )
        if video:
            video_kwargs = {
                "video_folder": os.path.join(self.work_dir, "videos", "train"),
                "step_trigger": lambda step: step % self.cfg.video_interval == 0,
                "video_length": self.cfg.video_length,
                "disable_logger": True,
            }
            print("[INFO]: Recording videos during training.")
            env = gym.wrappers.RecordVideo(env, **video_kwargs)
        self.env = MultiEnvWrapper(env)

    def initialize_agent(self):
        self.cfg.agent.obs_dim = self.env.observation_space.shape[0]
        self.cfg.agent.action_dim = self.env.action_space.shape[0]
        self.cfg.agent.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max()),
        ]
        self.agent = hydra.utils.instantiate(self.cfg.agent)

    def initialize_replay_buffer(self):
        self.replay_buffer = ReplayBuffer(
            obs_shape=self.env.observation_space.shape,
            action_shape=self.env.action_space.shape,
            capacity=int(self.cfg.replay_buffer_capacity),
            device=self.device,
            window=self.num_envs,
        )

    def initialize_combined_replay_buffer(self):
        self.replay_buffer = ReplayBuffer(
            obs_shape=self.env.observation_space.shape,
            action_shape=self.env.action_space.shape,
            capacity=int(self.cfg.replay_buffer_capacity),
            device=self.device,
            window=self.num_envs,
            alpha=self.cfg.reward_alpha,
            beta=self.cfg.reward_beta,
        )

    def initialize_reward_model_scriptedT(self):
        self.reward_model = reward_model.reward_model_scriptedT.RewardModel(
            ds=self.env.observation_space.shape[0],
            da=self.env.action_space.shape[0],
            ensemble_size=self.cfg.ensemble_size,
            lr=self.cfg.reward_lr,
            mb_size=self.cfg.reward_batch,
            size_segment=self.cfg.segment,
            activation=self.cfg.activation,
            large_batch=self.cfg.large_batch,
        )

    def initialize_reward_model_humanT(self):
        self.reward_model = reward_model.reward_model_humanT.RewardModel(
            dt=self.env.unwrapped.step_dt,
            ds=self.env.observation_space.shape[0],
            da=self.env.action_space.shape[0],
            ensemble_size=self.cfg.ensemble_size,
            lr=self.cfg.reward_lr,
            mb_size=self.cfg.reward_batch,
            size_segment=self.cfg.segment,
            activation=self.cfg.activation,
            large_batch=self.cfg.large_batch,
            near_range=self.cfg.near_range,
            env=self.env,
        )

    def learn_reward(self, first_flag=0):
        print()
        # get feedbacks
        labeled_queries = 0
        if first_flag == 1:
            # if it is first time to get feedback, need to use random sampling
            labeled_queries = self.reward_model.uniform_sampling()
        else:
            if self.cfg.feed_type == 0:
                labeled_queries = self.reward_model.uniform_sampling()
            elif self.cfg.feed_type == 1:
                labeled_queries = self.reward_model.disagreement_sampling()
            elif self.cfg.feed_type == 2:
                labeled_queries = (
                    self.reward_model.near_on_policy_disagreement_sampling()
                )
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

    def initialize_running(self):
        self.episode = 0
        self.env_episode_reward = np.zeros(self.num_envs)
        self.episode_done = np.zeros(self.num_envs)

        self.obs, self.body_state = self.env.reset()
        self.obs_np = self.obs.detach().cpu().numpy()
        self.body_state_np = self.body_state.detach().cpu().numpy()

    def handle_done(self, done_idx):
        self.obs[done_idx], self.body_state[done_idx] = self.env.get_obs(done_idx)
        self.obs_np = self.obs.detach().cpu().numpy()
        self.body_state_np = self.body_state.detach().cpu().numpy()

        self.logger.log(
            "train/episode_reward",
            self.env_episode_reward[done_idx].sum() / done_idx.size,
            self.step,
        )
        self.env_episode_reward[done_idx] = 0
        self.episode += done_idx.size
        self.logger.log("train/episode", self.episode, self.step)

    def set_random_action(self):
        self.action = torch.tensor(
            np.array([self.env.action_space.sample() for _ in range(self.num_envs)]),
            dtype=torch.float32,
            device=self.device,
        )
        self.action_np = self.action.detach().cpu().numpy()

    def set_agent_action(self):
        with my_utils.eval_mode(self.agent):
            self.action = self.agent.act(self.obs, sample=True)
        self.action_np = self.action.detach().cpu().numpy()

    def update_agent_during_unsupervised_pretraining(self):
        self.agent.update_state_ent(
            self.replay_buffer,
            self.logger,
            self.step,
            gradient_update=self.cfg.num_gradient_update,
            K=self.cfg.topK,
        )

    def update_agent_after_unsupervised_pretraining(self):
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

    def update_agent(self):
        self.agent.update(
            replay_buffer=self.replay_buffer,
            logger=self.logger,
            step=self.step,
            gradient_update=self.cfg.num_gradient_update,
        )

    def environment_step(self):
        (
            self.next_obs,
            self.reward,
            self.done,
            self.done_no_max,
            self.next_body_state,
        ) = self.env.step(self.action)

        self.next_obs_np = self.next_obs.detach().cpu().numpy()
        self.reward_np = self.reward.detach().cpu().numpy()
        self.done_np = self.done.float().detach().cpu().numpy()
        self.done_no_max_np = self.done_no_max.float().detach().cpu().numpy()
        self.next_body_state_np = self.next_body_state.detach().cpu().numpy()

    def reallocate_datas(self):
        self.episode_done = self.done_np
        self.env_episode_reward += self.reward_np

        self.obs = self.next_obs
        self.obs_np = self.next_obs_np
        self.body_state = self.next_body_state
        self.body_state_np = self.next_body_state_np
