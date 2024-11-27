import gymnasium as gym

from .walker2d_env import Walker2dEnv, Walker2dEnvCamera, Walker2dEnvCfg


def walker2d_env_create(
    seed=None, device="cuda:0", num_envs=1, render_mode=None, **kwargs
):
    cfg = Walker2dEnvCfg()
    cfg.seed = seed
    cfg.sim.device = device
    cfg.scene.num_envs = num_envs
    return Walker2dEnv(cfg=cfg, render_mode=render_mode, **kwargs)


def walker2d_env_camera_create(
    seed=None, device="cuda:0", num_envs=1, render_mode=None, **kwargs
):
    cfg = Walker2dEnvCfg()
    cfg.seed = seed
    cfg.sim.device = device
    cfg.scene.num_envs = num_envs
    return Walker2dEnvCamera(cfg=cfg, render_mode=render_mode, **kwargs)


gym.register(
    id="walker2d",
    entry_point=walker2d_env_create,
    disable_env_checker=True,
)

gym.register(
    id="walker2d_camera",
    entry_point=walker2d_env_camera_create,
    disable_env_checker=True,
)
