import gymnasium as gym

from .humanoid_env import HumanoidEnv, HumanoidEnvCfg


def humanoid_env_create(
    seed=None, device="cuda:0", num_envs=1, render_mode=None, **kwargs
):
    cfg = HumanoidEnvCfg()
    cfg.seed = seed
    cfg.sim.device = device
    cfg.scene.num_envs = num_envs
    return HumanoidEnv(cfg=cfg, render_mode=render_mode, **kwargs)


gym.register(
    id="humanoid",
    entry_point=humanoid_env_create,
    disable_env_checker=True,
)
