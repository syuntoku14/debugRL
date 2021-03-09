**Status:** Under development (expect bug fixes and huge updates)

# ShinRL: A python library for analyzing reinforcement learning

`ShinRL` is a reinforcement learning (RL) library with useful analysis tools.
It allows you to analyze the *shin* (*shin* means oracle in Japanese) behaviors of RL.

```python
import gym
import numpy as np
from shinrl.solvers.sac.discrete import SacSolver

env = gym.make("TabularPendulum-v0")
solver = SacSolver(env)
solver.run(num_steps=500)
```

![Ant](assets/ant.gif)
![Pendulum](assets/pendulum.gif)
![Tabular](assets/tabular.gif)

See [quickstart.py](examples/quickstart.py) and [tutorial.ipynb](examples/tutorial.ipynb) for the basic usages.

## Key features

### :zap: Oracle analysis with TabularEnv
* A flexible class `TabularEnv` provides useful functions for RL analysis. For example, you can calculate the oracle action-values with ``compute_action_values`` method.
* Subclasses of `TabularEnv` can be used as the regular OpenAI-Gym environments.
* Some environments support continuous action space and image observation.

#### Implemented environments

|                   Environment                    |   Dicrete action   | Continuous action  | Image Observation  | Tuple Observation  |
| :----------------------------------------------: | :----------------: | :----------------: | :----------------: | :----------------: |
|        [GridCraft](shinrl/envs/gridcraft)        | :heavy_check_mark: |        :x:         |        :x:         | :heavy_check_mark: |
| [TabularMountainCar-v0](shinrl/envs/mountaincar) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
|    [TabularPendulum-v0](shinrl/envs/pendulum)    | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
|    [TabularCartPole-v0](shinrl/envs/cartpole)    | :heavy_check_mark: | :heavy_check_mark: |        :x:         | :heavy_check_mark: |

See [shinrl/\_\_init\_\_.py](shinrl/__init__.py) for the available environments.

### :fire: Gym solvers
* `ShinRL` provides algorithms to solve the OpenAI-Gym environments as `Solver`.
* The dependencies between solvers are minimized to facilitate the addition and modification of new algorithms.
* Some solvers support the regular Gym environments as well as `TabularEnv`.
* Easy to visualize the training progress with [clearML](https://github.com/allegroai/clearml).

#### Implemented algorithms

|                                          Algorithms                                           |                 Discrete Control                  |                 Continuous Control                  |                                                                                  Solvers                                                                                   |
| :-------------------------------------------------------------------------------------------: | :-----------------------------------------------: | :-------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Value Iteration ([DQN](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)) | [:heavy_check_mark:](shinrl/solvers/vi/discrete)  |                         :x:                         | <sup id="a1">[1](#f1)</sup>OracleViSolver<br>SamplingViSolver<br>ExactFittedViSolver<br><sup id="a1">[2](#f2)</sup><span style="color:green">SamplingFittedViSolver</span> |
|        [Conservative Value Iteration](http://proceedings.mlr.press/v89/kozuno19a.html)        | [:heavy_check_mark:](shinrl/solvers/vi/discrete)  |                         :x:                         |                          OracleCviSolver<br>SamplingCviSolver<br>ExactFittedCviSolver<br><span style="color:green">SamplingFittedCviSolver</span>                          |
|                               Policy Gradient (REINFORCE, A2C)                                | [:heavy_check_mark:](shinrl/solvers/pg/discrete)  |                         :x:                         |                                                     ExactPgSolver<br><span style="color:green">SamplingPgSolver</span>                                                     |
|               [Interpolated Policy Gradient](https://arxiv.org/abs/1706.00387)                | [:heavy_check_mark:](shinrl/solvers/ipg/discrete) |                         :x:                         |                                                                 <span style="color:green">IpgSolver</span>                                                                 |
|                 [Proximal Policy Gradient](https://arxiv.org/abs/1707.06347)                  | [:heavy_check_mark:](shinrl/solvers/ppo/discrete) |                         :x:                         |                                                                 <span style="color:green">PpoSolver</span>                                                                 |
|                      [Soft Actor-Critic](shinrl/solvers/sac_continuous)                       | [:heavy_check_mark:](shinrl/solvers/sac/discrete) | [:heavy_check_mark:](shinrl/solvers/sac/continuous) |                                                                 <span style="color:green">SacSolver</span>                                                                 |

<b id="f1">1</b> The naming rule follows [Diagnosing Bottlenecks in Deep Q-learning Algorithms](https://arxiv.org/abs/1902.10250): 
* *Oracle-* solvers don't contain any errors. 
* *Sampling-* solvers use data sampled from MDP.
* *Exact Fitted-* solvers use function approximation but don't use sampled data.
* *Sampling Fitted-* solvers use both function approximation and sampled data. 

<b id="f2">2</b> Solvers colored with green support both TabularEnv and regular Gym environments.

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
