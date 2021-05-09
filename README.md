**Status:** Under development (expect bug fixes and huge updates)

# ShinRL: A python library for analyzing reinforcement learning

`ShinRL` is a reinforcement learning (RL) library with helpful analysis tools.
It allows you to analyze the *shin* (*shin* means oracle in Japanese) behaviors of RL.

```python
import gym
from shinrl.solvers.sac.discrete import SacSolver

env = gym.make("TabularPendulum-v0")
solver = SacSolver(env)

# train SAC
solver.run(num_steps=500)

# plot learned state values
q_values = solver.tb_values
v_values = np.sum(solver.tb_policy*q_values, axis=-1)
env.plot_values(v_values, title="State Values")
```

See [quickstart.py](experiments/tutorial/quickstart.py) and [tutorial.ipynb](experiments/tutorial/tutorial.ipynb) for the basic usages.
For more information, you can refer to [ShinRL's documentation](https://shinrl.readthedocs.io/en/latest/?).

![Ant](assets/ant.gif)
![Pendulum](assets/pendulum.gif)
![Tabular](assets/tabular.gif)


# :pencil: Experiments

The goal of ShinRL is not to provide state-of-the-art agents but to analyze the behaviors of RL algorithms.
To the end, ShinRL provides `experiments` that reproduce various analyses from many studies.
For example, ``experiments/VI_Error_Propagation`` analyzes the effects of entropy and KL regularization on the error tolerance and convergence speed of VI.

|                        Experiment                        |                       Objective                        |                                              Papers                                              |
| :------------------------------------------------------: | :----------------------------------------------------: | :----------------------------------------------------------------------------------------------: |
|             [Tutorial](experiments/Tutorial)             |                Learn how to use ShinRL                 |                                                                                                  |
| [VI_Performance_Bound](experiments/VI_Performance_Bound) | Examine the performance bound of various VI algorithms | [Leverage the Average: an Analysis of KL Regularization in RL](https://arxiv.org/abs/2003.14089) |


# :zap: Key features

## Oracle analysis with TabularEnv
* A flexible class `TabularEnv` provides useful functions for RL analysis. For example, you can calculate the oracle action-values with ``compute_action_values`` method.
* Subclasses of `TabularEnv` can be used as the regular OpenAI-Gym environments.
* Some environments support continuous action space and image observation.

|                   Environment                    |   Dicrete action   | Continuous action  | Image Observation  | Tuple Observation  |
| :----------------------------------------------: | :----------------: | :----------------: | :----------------: | :----------------: |
|        [GridCraft](shinrl/envs/gridcraft)        | :heavy_check_mark: |        :x:         |        :x:         | :heavy_check_mark: |
| [TabularMountainCar-v0](shinrl/envs/mountaincar) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
|    [TabularPendulum-v0](shinrl/envs/pendulum)    | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
|    [TabularCartPole-v0](shinrl/envs/cartpole)    | :heavy_check_mark: | :heavy_check_mark: |        :x:         | :heavy_check_mark: |

See [shinrl/\_\_init\_\_.py](shinrl/__init__.py) for the available environments.

## Gym solvers
* `ShinRL` provides algorithms to solve MDPs as `Solver`.
* Some solvers support the regular Gym environments as well as `TabularEnv`.
* The dependencies between solvers are minimized to facilitate the addition and modification of new algorithms.
* Easy to visualize the training progress with [ClearML](https://github.com/allegroai/clearml).

|                                                                                                                                       Algorithms                                                                                                                                        |                  Discrete Control                  |                  Continuous Control                  |                                                                            Solvers                                                                             |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------: | :--------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Value Iteration <br>([Deep Q Network (DQN)](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf), <br>[Conservative Value Iteration (CVI)](http://proceedings.mlr.press/v89/kozuno19a.html), <br>[Munchausen Value Iteration (MVI)](https://arxiv.org/abs/2007.14430)) |  [:heavy_check_mark:](shinrl/solvers/vi/discrete)  |                         :x:                          | <sup id="a1">[1](#f1)</sup>Oracle(C,M)ViSolver<br>Sampling(C,M)ViSolver<br>ExactFitted(C,M)ViSolver<br><sup id="a1">[2](#f2)</sup>SamplingFitted(C, M)ViSolver |
|                                                                                                 On-Policy Policy Gradient <br>(REINFORCE, A2C, [PPO](https://arxiv.org/abs/1707.06347))                                                                                                 | [:heavy_check_mark:](shinrl/solvers/onpg/discrete) | [:heavy_check_mark:](shinrl/solvers/onpg/continuous) |                              ExactPgSolver<br><sup id="a1">[2](#f2)</sup>SamplingPgSolver<br><sup id="a1">[2](#f2)</sup>PpoSolver                              |
|                                                                                                         [Interpolated Policy Gradient (IPG)](https://arxiv.org/abs/1706.00387)                                                                                                          | [:heavy_check_mark:](shinrl/solvers/ipg/discrete)  |                         :x:                          |                                                              <sup id="a1">[2](#f2)</sup>IpgSolver                                                              |
|                                                                                                                [Soft Actor-Critic (SAC)](shinrl/solvers/sac_continuous)                                                                                                                 | [:heavy_check_mark:](shinrl/solvers/sac/discrete)  | [:heavy_check_mark:](shinrl/solvers/sac/continuous)  |                                                              <sup id="a1">[2](#f2)</sup>SacSolver                                                              |

<b id="f1">1</b> The naming rule follows [Diagnosing Bottlenecks in Deep Q-learning Algorithms](https://arxiv.org/abs/1902.10250): 
* *Oracle-* solvers don't contain any errors. 
* *Sampling-* solvers use data sampled from MDP.
* *Exact Fitted-* solvers use function approximation but don't use sampled data.
* *Sampling Fitted-* solvers use both function approximation and sampled data. 

<b id="f2">2</b> Those solvers support both TabularEnv and regular Gym environments.

# Installation

```bash
git clone git@github.com:syuntoku14/ShinRL.git
cd ShinRL
pip install -e .
```

# Citation

```
@misc{toshinori2020shinrl,
    author = {Toshinori Kitamura},
    title = {ShinRL},
    year = {2020},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/syuntoku14/ShinRL}}
}
```
