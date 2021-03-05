import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from shinrl.envs.pendulum import Pendulum, plot_pendulum_values, reshape_values
from shinrl.solvers import SacSolver
device = "cuda" if torch.cuda.is_available() else "cpu"


env = Pendulum()
solver = SacSolver(env, solve_options={"device": device})

# Step 1: train networks
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
