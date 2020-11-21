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
現在は以下の環境をサポートしています:

| Environment | Dicrete action | Continuous action | Image Observation | Tuple Observation |
| :-----: | :-----: | :-----: | :-----: | :-----: |
| [GridCraft](debug_rl/envs/gridcraft) | ✓ | - | - | ✓ |
| [MountainCar](debug_rl/envs/mountaincar) | ✓ | ✓ | ✓ | ✓ |
| [Pendulum](debug_rl/envs/pendulum) | ✓ | ✓ | ✓ | ✓ |
| [CartPole](debug_rl/envs/cartpole) | ✓ | ✓ | - | ✓ |


* `solvers`: Solverはenvとソルバのハイパーパラメータである`options`を初期化時に設定し, `solve`関数を実行することでMDPを解きます. 
また, 初期化時にtrainsのloggerを渡すことで[trains](https://github.com/allegroai/trains)による学習の確認もできますが, 渡さなくても実行は可能です.
真の累積報酬を計算する``compute_expected_return``や真の行動価値観数を計算する``compute_action_values``などの便利な関数は[debug_rl/solvers/base.py](debug_rl/solvers/base.py)を参照してください.
現在は以下のソルバをサポートしています:

| Solver | Sample approximation | Function approximation | Continuous Action | Algorithm |
| :---:| :---: | :---: | :---: | :---: |
| [OracleViSolver, OracleCviSolver](debug_rl/solvers/oracle_vi) | - | - | - | Q-learning, [Conservative Value Iteration (CVI)](http://proceedings.mlr.press/v89/kozuno19a.html) |
| [ExactFittedViSolver, ExactFittedCviSolver](debug_rl/solvers/exact_fvi) | - | ✓ | - | Fitted Q-learning, Fitted CVI |
| [SamplingViSolver, SamplingCviSolver](debug_rl/solvers/sampling_vi) | ✓ | - | - | Q-learning, CVI |
| [SamplingFittedViSolver, SamplingFittedCviSolver](debug_rl/solvers/sampling_fvi) | ✓ | ✓ | - | Fitted Q-learning, Fitted CVI |
| [SacSolver](debug_rl/solvers/sac) | ✓ | ✓ | - | [Discrete Soft Actor Critic](https://arxiv.org/abs/1910.07207) |
| [SacContinuousSolver](debug_rl/solvers/sac_continuous) | ✓ | ✓ | ✓ | [Soft Actor Critic](https://arxiv.org/abs/1801.01290) |

# インストール

```bash
$ git clone git@github.com:syuntoku14/debugRL.git
$ cd debugRL
$ pip install -e .
```