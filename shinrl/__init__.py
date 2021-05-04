from gym.envs.registration import register

register(
    id="TabularPendulum-v0",
    entry_point="shinrl.envs:Pendulum",
    kwargs={
        "horizon": 200,
        "action_mode": "discrete",
        "dA": 5,
        "state_disc": 32,
        "obs_mode": "tuple",
    },
)

register(
    id="TabularPendulumContinuous-v0",
    entry_point="shinrl.envs:Pendulum",
    kwargs={
        "horizon": 200,
        "action_mode": "continuous",
        "dA": 50,
        "state_disc": 32,
        "obs_mode": "tuple",
    },
)

register(
    id="TabularPendulumImage-v0",
    entry_point="shinrl.envs:Pendulum",
    kwargs={
        "horizon": 200,
        "action_mode": "discrete",
        "dA": 5,
        "state_disc": 32,
        "obs_mode": "image",
    },
)

register(
    id="TabularPendulumImageContinuous-v0",
    entry_point="shinrl.envs:Pendulum",
    kwargs={
        "horizon": 200,
        "action_mode": "continuous",
        "dA": 50,
        "state_disc": 32,
        "obs_mode": "image",
    },
)

register(
    id="TabularMountainCar-v0",
    entry_point="shinrl.envs:MountainCar",
    kwargs={
        "horizon": 200,
        "action_mode": "discrete",
        "dA": 5,
        "state_disc": 32,
        "obs_mode": "tuple",
    },
)

register(
    id="TabularMountainCarContinuous-v0",
    entry_point="shinrl.envs:MountainCar",
    kwargs={
        "horizon": 200,
        "action_mode": "continuous",
        "dA": 50,
        "state_disc": 32,
        "obs_mode": "tuple",
    },
)

register(
    id="TabularMountainCarImage-v0",
    entry_point="shinrl.envs:MountainCar",
    kwargs={
        "horizon": 200,
        "action_mode": "discrete",
        "dA": 5,
        "state_disc": 32,
        "obs_mode": "image",
    },
)

register(
    id="TabularMountainCarImageContinuous-v0",
    entry_point="shinrl.envs:MountainCar",
    kwargs={
        "horizon": 200,
        "action_mode": "continuous",
        "dA": 50,
        "state_disc": 32,
        "obs_mode": "image",
    },
)

register(
    id="TabularCartPole-v0",
    entry_point="shinrl.envs:CartPole",
    kwargs={
        "horizon": 100,
        "action_mode": "discrete",
        "dA": 5,
        "x_disc": 128,
        "x_dot_disc": 4,
        "th_disc": 64,
        "th_dot_disc": 4,
        "obs_mode": "tuple",
    },
)
