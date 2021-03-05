from gym.envs.registration import register


register(
    id='ShinPendulum-v0',
    entry_point='shinrl.envs.pendulum:Pendulum',
    max_episode_steps=200,
)

register(
    id='ShinMountainCar-v0',
    entry_point='shinrl.envs.mountaincar:MountainCar',
    max_episode_steps=200,
)

register(
    id='ShinCartPole-v0',
    entry_point='shinrl.envs.cartpole:CartPole',
    max_episode_steps=100,
)
