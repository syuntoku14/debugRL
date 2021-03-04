from gym.envs.registration import register

# Algorithmic
# ----------------------------------------

register(
    id='RLowanPendulum-v0',
    entry_point='rlowan.envs.pendulum:Pendulum',
    max_episode_steps=200,
)

register(
    id='RLowanMountainCar-v0',
    entry_point='rlowan.envs.mountaincar:MountainCar',
    max_episode_steps=200,
)

register(
    id='RLowanCartPole-v0',
    entry_point='rlowan.envs.cartpole:CartPole',
    max_episode_steps=100,
)
