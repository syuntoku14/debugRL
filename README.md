**Status:** Under development (expect bug fixes and huge updates)

# ShinRL: A python library for analyzing reinforcement learning

`ShinRL` is a reinforcement learning (RL) library with useful analysis tools.
It allows you to analyze the *shin* (*shin* means oracle in Japanese) behaviors of RL.

## Key features

### :zap: Oracle analysis with TabularEnv
* A flexible class `TabularEnv` provides useful functions for RL analysis. For example, you can calculate the oracle action-values with ``compute_action_values`` method.
* Subclasses of `TabularEnv` can be used as the regular OpenAI-Gym environments.
* Some environments support continuous action space and image observation.

#### Implemented environments

|               Environment                |   Dicrete action   | Continuous action  | Image Observation  | Tuple Observation  |
| :--------------------------------------: | :----------------: | :----------------: | :----------------: | :----------------: |
|   [GridCraft](shinrl/envs/gridcraft)   | :heavy_check_mark: |        :x:         |        :x:         | :heavy_check_mark: |
| [MountainCar](shinrl/envs/mountaincar) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
|    [Pendulum](shinrl/envs/pendulum)    | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
|    [CartPole](shinrl/envs/cartpole)    | :heavy_check_mark: | :heavy_check_mark: |        :x:         | :heavy_check_mark: |

### :fire: Practical MDP solvers
* `ShinRL` provides algorithms to solve the OpenAI-Gym environments as `Solver`.
* The dependencies between solvers are minimized to facilitate the addition and modification of new algorithms.
* Some solvers support the regular Gym environments as well as `TabularEnv`.
* Easy to visualize the training progress with [clearML](https://github.com/allegroai/clearml).

#### Implemented solvers

|                                      Solver                                      | Sample approximation | Function approximation | Continuous Action  |                                                  Algorithm                                                  |
| :------------------------------------------------------------------------------: | :------------------: | :--------------------: | :----------------: | :---------------------------------------------------------------------------------------------------------: |
|          [OracleViSolver, OracleCviSolver](shinrl/solvers/oracle_vi)           |         :x:          |          :x:           |        :x:         |      Q-learning, [Conservative Value Iteration (CVI)](http://proceedings.mlr.press/v89/kozuno19a.html)      |
|     [ExactFittedViSolver, ExactFittedCviSolver](shinrl/solvers/exact_fvi)      |         :x:          |   :heavy_check_mark:   |        :x:         |                                        Fitted Q-learning, Fitted CVI                                        |
|       [SamplingViSolver, SamplingCviSolver](shinrl/solvers/sampling_vi)        |  :heavy_check_mark:  |          :x:           |        :x:         |                                               Q-learning, CVI                                               |
| [SamplingFittedViSolver, SamplingFittedCviSolver](shinrl/solvers/sampling_fvi) |  :heavy_check_mark:  |   :heavy_check_mark:   |        :x:         | Fitted Q-learning ([DQN](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)), Fitted CVI |
|                    [ExactPgSolver](shinrl/solvers/exact_pg)                    |         :x:          |   :heavy_check_mark:   |        :x:         |                                               Policy gradient                                               |
|                 [SamplingPgSolver](shinrl/solvers/sampling_pg)                 |         :x:          |   :heavy_check_mark:   |        :x:         |                                      Policy gradient (REINFORCE, A2C)                                       |
|                        [IpgSolver](shinrl/solvers/ipg)                         |         :x:          |   :heavy_check_mark:   |        :x:         |                      [Interpolated policy gradient](https://arxiv.org/abs/1706.00387)                       |
|                        [SacSolver](shinrl/solvers/sac)                         |  :heavy_check_mark:  |   :heavy_check_mark:   |        :x:         |                       [Discrete Soft Actor Critic](https://arxiv.org/abs/1910.07207)                        |
|              [SacContinuousSolver](shinrl/solvers/sac_continuous)              |  :heavy_check_mark:  |   :heavy_check_mark:   | :heavy_check_mark: |                            [Soft Actor Critic](https://arxiv.org/abs/1801.01290)                            |
|                        [PpoSolver](shinrl/solvers/ppo)                         |  :heavy_check_mark:  |   :heavy_check_mark:   |        :x:         |                 [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)                 |

# Getting started

Here, we see a simple debugging example using shinrl.
The described code can be executed by:

```bash
python examples/simple.py
```

TabularEnv has several methods to compute oracle values.
Using those methods, you can analyze whether trained models actually solve the MDP or not.

* ```env.compute_action_values(policy)``` returns the oracle Q values from a policy matrix (numpy.array with `# of states`x`# of actions`).
* ```env.compute_visitation(policy, discount=1.0)``` returns the oracle normalized discounted stationary distribution from a policy matrix.
* ```env.compute_expected_return(policy)``` returns the oracle cumulative rewards from a policy matrix.

In this example, we train a SAC model in Pendulum environment, and check whether the model successfully learns the soft Q values using ```compute_action_values``` function.
Since Pendulum environment can plot only V values instead Q values, out goal is to plot trained soft V values.

We do debugging as follows:

1. Train a model. We use the SAC implementation from shinrl in this example. You can use any models as long as it returns Q values or action probabilities from observations.
2. Using all_observations from TabularEnv, compute the policy matrix.
3. Compute soft Q values by env.compute_action_values. Since Pendulum environment supports only V values plotting, the following code plots V values instead. Check GridCraft environment if you want to see the behavior of Q values (see [examples/tutorial.ipynb](examples/tutorial.ipynb) for details).

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from shinrl.envs.pendulum import Pendulum, plot_pendulum_values, reshape_values
from shinrl.solvers import SacSolver
device = "cuda" if torch.cuda.is_available() else "cpu"

# Step 1: train networks
env = Pendulum()
solver = SacSolver(env, solve_options={"device": device})
solver.run(num_steps=5000)
value_network = solver.value_network
policy_network = solver.policy_network

# Step 2: create policy matrix
tensor_all_obss = torch.tensor(
    env.all_observations, dtype=torch.float32, device=device)
preference = policy_network(tensor_all_obss).reshape(
    env.dS, env.dA).detach().cpu().numpy()  # dS x dA
policy = special.softmax(preference, axis=-1).astype(np.float64)
policy /= policy.sum(axis=-1, keepdims=True)  # dS x dA

# Step 3: plot soft Q values
oracle_Q = env.compute_action_values(
    policy, er_coef=solver.solve_options["er_coef"])  # dS x dA
trained_Q = value_network(tensor_all_obss).reshape(
    env.dS, env.dA).detach().cpu().numpy()  # dS x dA
V_max = max(oracle_Q.max(), trained_Q.max())
V_min = min(oracle_Q.min(), trained_Q.min())

oracle_V = np.sum(policy*oracle_Q, axis=-1)
oracle_V = reshape_values(env, oracle_V)  # angles x velocities
print("Press Q on the image to go next.")
plot_pendulum_values(env, oracle_V, vmin=V_min,
                     vmax=V_max, title="Oracle State values: t=0")
plt.show()

trained_V = np.sum(policy*trained_Q, axis=-1)
trained_V = reshape_values(env, trained_V)  # angles x velocities
plot_pendulum_values(env, trained_V, vmin=V_min,
                     vmax=V_max, title="Trained State values: t=0")
plt.show()
```

The code above will generate the following figures.
The upper figure shows the oracle soft V values, and the bottom figure shows the trained soft V values.

![](assets/oracle_V.png)
![](assets/trained_V.png)

Since the oracle and the trained V values are quite similar, we can conclude that the network learns the soft Q values successfully.

# Installation

```bash
git clone git@github.com:syuntoku14/ShinRL.git
cd ShinRL
pip install -e .
```

# Citation

```
@misc{toshinori2020shinrl,
    author = {Kitamura Toshinori},
    title = {ShinRL},
    year = {2020},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/syuntoku14/ShinRL}}
}
```
