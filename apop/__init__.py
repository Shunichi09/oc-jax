__version__ = "0.0.1"

from gymnasium.envs.registration import register

register(
    id="Arm2dReachingTargetFixed-v0",
    entry_point="apop.envs.mujoco.arm2d_reaching:Arm2dReachingTargetFixedEnv",
    max_episode_steps=5000,
)

register(
    id="ContinuousInvertedCartPole-v0",
    entry_point="apop.envs.gymnasium.cartpole:ContinuousInvertedCartPoleEnv",
    max_episode_steps=500,
)
