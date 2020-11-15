# DebugRL: A python library for debugging RL algorithms

[日本語](assets/README.jp.md) | English


`debug_rl` is a library for analyzing the behavior of reinforcement learning algorithms, including DeepRL. 
This library is inspired by [diag_q](https://github.com/justinjfu/diagnosing_qlearning) (See [Diagnosing Bottlenecks in Deep Q-learning Algorithms](https://arxiv.org/abs/1902.10250)), but has been developed with an emphasis on ease of use.
Using matrix-form representation of the dynamics of environments, debug_rl allows you to calculate the oracle values such as action-value functions and stationary distributions that can only be approximated by multiple samples in the usual Gym environment.
To facilitate the addition and modification of new algorithms, implementations follows [OpenAI Spinningup](https://github.com/openai/spinningup)-style to minimize the dependencies between the algorithms.
See the notebook in [examples](examples) for basic usage.

The debug_rl is constructed of two parts: the `envs` and the `solvers`.

* `envs`: All environments are subclasses of [TabularEnv](debug_rl/envs/base.py), but have step and reset functions similar to those of OpenAI Gym, so they can be used in the same way as the regular Gym environment.
Some of the environments support continuous action input and image-based observation modes that enable analysis of continuous action RL and CNN.
Currently, the following environments are supported:

| Environment | Dicrete action | Continuous action | Image Observation | Tuple Observation |
| :----------------------------------------------------------: | :------------: | :---------------: | :----------------------------------------: | ------------------------ |
| [GridCraft](debug_rl/envs/gridcraft) | ✓ | - | - | ✓ |
| [MountainCar](debug_rl/envs/mountaincar) | ✓ | ✓ | ✓ | ✓ |
| [Pendulum](debug_rl/envs/pendulum) | ✓ | ✓ | ✓ | ✓ |
| [CartPole](debug_rl/envs/cartpole) | ✓ | ✓ | - | ✓ |

* `solvers`: A solver solves MDP by setting the env and the hyperparameters at initialization and then executing the `solve` function. 
You can also check the training progress by passing the 
logger of [trains](https://github.com/allegroai/trains) at the initialization, but you can still run the solver without it.
See [debug_rl/solvers/base.py](debug_rl/solvers/base.py) for useful functions such as ``compute_expected_return`` to calculate true cumulative rewards and ``compute_action_values`` to calculate true action values.
Currently the following solvers are supported:

| Solver | Sample approximation | Function approximation | Continuous Action | Algorithm |
| :---:| :---: | :---: | :---: | :---: |
| [OracleViSolver, OracleCviSolver](debug_rl/solvers/oracle_vi) | - | - | - | Q-learning, [Conservative Value Iteration (CVI)](http://proceedings.mlr.press/v89/kozuno19a.html) |
| [ExactFittedViSolver, ExactFittedCviSolver](debug_rl/solvers/exact_fvi) | - | ✓ | - | Fitted Q-learning, Fitted CVI |
| [SamplingViSolver, SamplingCviSolver](debug_rl/solvers/sampling_vi) | ✓ | - | - | Q-learning, CVI |
| [SamplingFittedViSolver, SamplingFittedCviSolver](debug_rl/solvers/sampling_fvi) | ✓ | ✓ | - | Fitted Q-learning, Fitted CVI |
| [SacSolver](debug_rl/solvers/sac) | ✓ | ✓ | - | [Discrete Soft Actor Critic](https://arxiv.org/abs/1910.07207) |
| [SacContinuousSolver](debug_rl/solvers/sac_continuous) | ✓ | ✓ | ✓ | [Soft Actor Critic](https://arxiv.org/abs/1801.01290) |


```
@misc{toshinori2020debugrl,
    author = {Kitamura Toshinori},
    title = {DebugRL},
    year = {2020},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/syuntoku14/debugRL}}
}
```