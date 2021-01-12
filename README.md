**Status:** Under development (expect bug fixes and huge updates)

# DebugRL: A python library for debugging RL algorithms

[日本語](assets/README.jp.md) | English


`debug_rl` is a library for analyzing the behavior of reinforcement learning algorithms, including DeepRL. 
This library is inspired by [diag_q](https://github.com/justinjfu/diagnosing_qlearning) (See [Diagnosing Bottlenecks in Deep Q-learning Algorithms](https://arxiv.org/abs/1902.10250)), but has been developed with an emphasis on ease of use.
Using matrix-form representation of the dynamics of environments, debug_rl allows you to calculate the oracle values such as action-value functions and stationary distributions that can only be approximated by multiple samples in the usual Gym environment.
To facilitate the addition and modification of new algorithms, implementations follows [OpenAI Spinningup](https://github.com/openai/spinningup)-style to minimize the dependencies between the algorithms.
See the notebook in [examples](examples) for basic usage.

The debug_rl is constructed of two parts: the `envs` and the `solvers`.
Although the `solvers` requires PyTorch, it is possible to install only `envs` for users who do not use PyTorch (See [Installation](#Installation)).

* `envs`: All environments are subclasses of [TabularEnv](debug_rl/envs/base.py), but have step and reset functions similar to those of OpenAI Gym, so they can be used in the same way as the regular Gym environment.
Some of the environments support continuous action input and image-based observation modes that enable analysis of continuous action RL and CNN.
TabularEnv class has useful functions such as ``compute_expected_return`` to calculate true cumulative rewards and ``compute_action_values`` to calculate true action values (see [debug_rl/envs/base.py](debug_rl/envs/base.py) for details).
Currently, the following environments are supported:

| Environment | Dicrete action | Continuous action | Image Observation | Tuple Observation |
| :----------------------------------------------------------: | :------------: | :---------------: | :----------------------------------------: | ------------------------ |
| [GridCraft](debug_rl/envs/gridcraft) | ✓ | - | - | ✓ |
| [MountainCar](debug_rl/envs/mountaincar) | ✓ | ✓ | ✓ | ✓ |
| [Pendulum](debug_rl/envs/pendulum) | ✓ | ✓ | ✓ | ✓ |
| [CartPole](debug_rl/envs/cartpole) | ✓ | ✓ | - | ✓ |

* `solvers`: A solver solves MDP by setting the env and the hyperparameters at initialization and then executing the `solve` function. 
You can also check the training progress by passing the 
logger of [clearML](https://github.com/allegroai/clearml) at the initialization, but you can still run the solver without it.
Currently the following solvers are supported:

| Solver | Sample approximation | Function approximation | Continuous Action | Algorithm |
| :---:| :---: | :---: | :---: | :---: |
| [OracleViSolver, OracleCviSolver](debug_rl/solvers/oracle_vi) | - | - | - | Q-learning, [Conservative Value Iteration (CVI)](http://proceedings.mlr.press/v89/kozuno19a.html) |
| [ExactFittedViSolver, ExactFittedCviSolver](debug_rl/solvers/exact_fvi) | - | ✓ | - | Fitted Q-learning, Fitted CVI |
| [SamplingViSolver, SamplingCviSolver](debug_rl/solvers/sampling_vi) | ✓ | - | - | Q-learning, CVI |
| [SamplingFittedViSolver, SamplingFittedCviSolver](debug_rl/solvers/sampling_fvi) | ✓ | ✓ | - | Fitted Q-learning, Fitted CVI |
| [SacSolver](debug_rl/solvers/sac) | ✓ | ✓ | - | [Discrete Soft Actor Critic](https://arxiv.org/abs/1910.07207) |
| [SacContinuousSolver](debug_rl/solvers/sac_continuous) | ✓ | ✓ | ✓ | [Soft Actor Critic](https://arxiv.org/abs/1801.01290) |


# Getting started

Here, we see a simple debugging example using debug_rl.
The described code can be executed by:
```
python examples/simple_debug.py
```

TabularEnv has several methods to compute oracle values.
Using those methods, you can analyze whether trained models actually solve the MDP or not.

* ```env.compute_action_values(policy)``` returns the oracle Q values from a policy matrix (numpy.array with `# of states`x`# of actions`).
* ```env.compute_visitation(policy, discount=1.0)``` returns the oracle normalized discounted stationary distribution from a policy matrix.
* ```env.compute_expected_return(policy)``` returns the oracle cumulative rewards from a policy matrix.

In this example, we train a SAC model in Pendulum environment, and check whether the model successfully learns the Q values using ```compute_action_values``` function.
Since Pendulum environment can plot only V values instead Q values, we treat the maximum value of Q values as the V values. 

We do debugging as follows:

1. Train a model. We use the SAC implementation from debug_rl in this example. You can use any models as long as it returns Q values or action probabilities from observations.
2. Using all_observations from TabularEnv, compute the policy matrix.
3. Compute Q values by env.compute_action_values. Since Pendulum environment supports only V values plotting, the following code plots V values instead. Check GridCraft environment if you want to see the behavior of Q values (see [examples/tutorial.ipynb](examples/tutorial.ipynb) for details).

```
import torch
import numpy as np
import matplotlib.pyplot as plt
from debug_rl.envs.pendulum import Pendulum, plot_pendulum_values, reshape_values
from debug_rl.solvers import SacSolver
device = "cuda" if torch.cuda.is_available() else "cpu"

# Step 1: train networks
env = Pendulum()
solver = SacSolver(env, solve_options={"num_trains": 5000, "device": device})
solver.solve()
value_network = solver.value_network
policy_network = solver.policy_network

# Step 2: create policy matrix
tensor_all_obss = torch.tensor(
    env.all_observations, dtype=torch.float32, device=device)
policy = policy_network(tensor_all_obss).reshape(
    env.dS, env.dA).detach().cpu().numpy()  # dS x dA

# Step 3: plot Q values
oracle_Q = env.compute_action_values(policy)  # dS x dA
oracle_V = np.max(oracle_Q, axis=-1)
oracle_V = reshape_values(env, oracle_V)  # angles x velocities
print("Press Q on the image to go next.")
plot_pendulum_values(env, oracle_V, vmin=oracle_Q.min(),
                     vmax=oracle_Q.max(), title="Oracle State values: t=0")
plt.show()

trained_Q = value_network(tensor_all_obss).reshape(
    env.dS, env.dA).detach().cpu().numpy()  # dS x dA
trained_V = np.max(trained_Q, axis=-1)
trained_V = reshape_values(env, trained_V)  # angles x velocities
plot_pendulum_values(env, trained_V, vmin=trained_Q.min(),
                     vmax=trained_Q.max(), title="Trained State values: t=0")
plt.show()
```

The code above will generate the following figures.
The upper figure shows the oracle V values, and the bottom figure shows the trained V values.

![](assets/oracle_V.png)
![](assets/trained_V.png)


# Installation

You can install both the debug_rl.envs and debug_rl.solvers by:
```bash
$ git clone git@github.com:syuntoku14/debugRL.git
$ cd debugRL
$ pip install -e .[solver]
```

If you want to use only the debug_rl.envs, install debug_rl by:
```bash
$ git clone git@github.com:syuntoku14/debugRL.git
$ cd debugRL
$ pip install -e .
```

# Citation

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