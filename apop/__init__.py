__version__ = "0.0.1"

from gymnasium.envs.registration import register

register(
    id="Arm2dReachingTargetFixed-v0",
    entry_point="apop.envs.mujoco.arm2d_reaching:Arm2dReachingTargetFixedEnv",
    max_episode_steps=5000,
)


register(
    id="Ball2dTracking-v0",
    entry_point="apop.envs.mujoco.ball2d_moving:Ball2dTrackingEnv",
    max_episode_steps=1000,
)

register(
    id="ContinuousInvertedCartPole-v0",
    entry_point="apop.envs.classic_control.cartpole:ContinuousInvertedCartPoleEnv",
    max_episode_steps=500,
)

register(
    id="ContinuousSwingUpCartPole-v0",
    entry_point="apop.envs.classic_control.cartpole:ContinuousSwingUpCartPoleEnv",
    max_episode_steps=500,
)

register(
    id="ContinuousAcrobot-v0",
    entry_point="apop.envs.classic_control.acrobot:ContinuousAcrobotEnv",
    max_episode_steps=1000,
)

register(
    id="AnglePendulum-v0",
    entry_point="apop.envs.classic_control.pendulum:AnglePendulumEnv",
    max_episode_steps=500,
)
