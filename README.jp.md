# DebugRL: 強化学習アルゴリズムのデバッグ用ライブラリ

日本語 | [English](README.md)

`debug_rl` はDeepRLを含んだ強化学習アルゴリズムの挙動解析をするために開発したライブラリです.
このライブラリは[diag_q](https://github.com/justinjfu/diagnosing_qlearning)を基に, より使いやすさを重視して開発しています (基本的な設計理念は[Diagnosing Bottlenecks in Deep Q-learning Algorithms](https://arxiv.org/abs/1902.10250)を参照してください).
debug_rlはGym形式の環境のダイナミクスを離散化, 行列形式で表現しており, 行動価値関数や定常分布などの`通常のGym環境では複数回のサンプルによって近似することでしか得られない真値`の計算ができるよう開発されています.
また, 新しいアルゴリズムを容易に追加・修正することができるよう, アルゴリズム同士の依存関係をなるべく少なくした[OpenAI Spinningup](https://github.com/openai/spinningup)形式の設計を意識しています.
**基本的な使い方は [examples](examples)にあるノートブックを参考にしてください.**

debug_rlは行列形式で表現された`envs`と, その`envs`を解く`solvers`の2つによって構築されています.

* `envs`: 全ての環境は[TabularEnv](debug_rl/envs/base.py)のサブクラスですが, OpenAI Gymと同様にstepやreset関数が備わっているため, 通常のGym環境と同様の使い方ができます. 
いくつかの環境では連続行動入力と画像形式の観測のモードを対応しており, 連続行動RLやCNNの解析も可能にしています.
真の累積報酬を計算する``compute_expected_return``や真の行動価値観数を計算する``compute_action_values``などの便利な関数は[debug_rl/envs/base.py](debug_rl/envs/base.py)を参照してください.
現在は以下の環境をサポートしています:

| Environment | Dicrete action | Continuous action | Image Observation | Tuple Observation |
| :-----: | :-----: | :-----: | :-----: | :-----: |
| [GridCraft](debug_rl/envs/gridcraft) | ✓ | - | - | ✓ |
| [MountainCar](debug_rl/envs/mountaincar) | ✓ | ✓ | ✓ | ✓ |
| [Pendulum](debug_rl/envs/pendulum) | ✓ | ✓ | ✓ | ✓ |
| [CartPole](debug_rl/envs/cartpole) | ✓ | ✓ | - | ✓ |


* `solvers`: Solverはenvとソルバのハイパーパラメータである`options`を初期化時に設定し, `run`関数を実行することでMDPを解きます. 
solverは`initialize`関数を呼ぶと初期化されます. `run`関数を繰り返し呼んで途中から学習を再開することも可能です.
また, 初期化時にclearMLのloggerを渡すことで[clearML](https://github.com/allegroai/clearml)による学習の確認もできますが, 渡さなくても実行は可能です.
現在は以下のソルバをサポートしています:

| Solver | Sample approximation | Function approximation | Continuous Action | Algorithm |
| :---:| :---: | :---: | :---: | :---: |
| [OracleViSolver, OracleCviSolver](debug_rl/solvers/oracle_vi) | - | - | - | Q-learning, [Conservative Value Iteration (CVI)](http://proceedings.mlr.press/v89/kozuno19a.html) |
| [ExactFittedViSolver, ExactFittedCviSolver](debug_rl/solvers/exact_fvi) | - | ✓ | - | Fitted Q-learning, Fitted CVI |
| [SamplingViSolver, SamplingCviSolver](debug_rl/solvers/sampling_vi) | ✓ | - | - | Q-learning, CVI |
| [SamplingFittedViSolver, SamplingFittedCviSolver](debug_rl/solvers/sampling_fvi) | ✓ | ✓ | - | Fitted Q-learning, Fitted CVI |
| [ExactPgSolver](debug_rl/solvers/exact_pg) | - | ✓ | - | Policy gradient |
| [SamplingPgSolver](debug_rl/solvers/sampling_pg) | - | ✓ | - | Policy gradient (REINFORCE, A2C)|
| [IpgSolver](debug_rl/solvers/ipg) | - | ✓ | - | [Interpolated policy gradient](https://arxiv.org/abs/1706.00387)|
| [SacSolver](debug_rl/solvers/sac) | ✓ | ✓ | - | [Discrete Soft Actor Critic](https://arxiv.org/abs/1910.07207) |
| [SacContinuousSolver](debug_rl/solvers/sac_continuous) | ✓ | ✓ | ✓ | [Soft Actor Critic](https://arxiv.org/abs/1801.01290) |
| [PpoSolver](debug_rl/solvers/ppo) | ✓ | ✓ | - | [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) |


# デバッグの流れ

以下で, 簡単な例通してdebug_rlを使ったデバッグの流れを見てみましょう. 
ここで紹介するコードは
```
python examples/simple_debug.py
```
で実行することができます.

TabularEnvには真の値を計算するためのモジュールが搭載されており, これを利用して学習したモデルが正しくMDPを解けているか確認できます.

* ```env.compute_action_values(policy)```関数は方策の行列 (`状態数`x`行動数`のnumpy.array) から真のQ値を計算します.
* ```env.compute_visitation(policy, discount=1.0)```関数は方策の行列 (`状態数`x`行動数`のnumpy.array)と割引率から正規化された割引定常分布を計算します.
* ```env.compute_expected_return(policy)```関数は方策の行列 (`状態数`x`行動数`のnumpy.array) から真の累積報酬和を計算します.

今回はPendulum環境でDQNを学習させ, ```compute_action_values```を使って学習したモデルが正しくsoft Q値を学習できているか確認します.
Pendulum環境ではQ値ではなく状態価値しか描画できないため, 今回はV値を描画してみましょう.

Q値のデバッグは以下のような流れで行います.

1. 学習したモデルを用意します. 今回はdebug_rlのSAC実装を使いますが, 観測に対してQ値や行動の確率を返すモデルであれば任意のモデルに簡単に拡張できます.
2. モデルに対して, TabularEnvのall_observations変数 (`状態数`x`観測の次元`)に格納されている全状態に対する観測をを利用して, 方策行列を回帰します.
3. env.compute_action_values関数を使ってモデルが学習したQ値を計算します. 今回はPendulum環境を使っているためV値を描画しますが, GridCraftではQ値をそのまま描画できます (詳しくは[examples/tutorial.ipynb](examples/tutorial.ipynb)を参照してください).

```
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from debug_rl.envs.pendulum import Pendulum, plot_pendulum_values, reshape_values
from debug_rl.solvers import SacSolver
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

実行すると以下の図が描画されます. 
上が真のV値, 下が学習されたV値です. 

真のsoft Q値とかなり似たものが回帰されているので, 正しくsoft Q関数を学習できていそうです.

![](assets/oracle_V.png)
![](assets/trained_V.png)

# インストール

```bash
$ git clone git@github.com:syuntoku14/debugRL.git
$ cd debugRL
$ pip install -e .
```