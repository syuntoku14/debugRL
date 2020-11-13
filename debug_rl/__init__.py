from gym.envs.registration import register

# Algorithmic
# ----------------------------------------

register(
    id='DebugPendulum-v0',
    entry_point='debug_rl.envs.pendulum:Pendulum',
    max_episode_steps=200,
)

register(
    id='DebugMountainCar-v0',
    entry_point='debug_rl.envs.mountaincar:MountainCar',
    max_episode_steps=200,
)

register(
    id='DebugCartPole-v0',
    entry_point='debug_rl.envs.cartpole:CartPole',
    max_episode_steps=100,
)
