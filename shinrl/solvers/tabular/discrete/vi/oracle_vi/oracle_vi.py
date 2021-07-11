import numpy as np
from scipy import special
from tqdm import tqdm

from shinrl import utils

from .core import Solver


class OracleSolver(Solver):
    """
    OracleSolver assumes full knowledge of MDP: the dynamics and the reward function.
    The value table is updated as:

    .. math::
        Q \\leftarrow Q+\\alpha K D_{\\rho}\\left(\\mathcal{T}^{*} Q-Q\\right),

    where we include several types of errors for advanced analysis: :math:`D_{\\rho}` is a visitation matrix for sampling errors and :math:`K` is a constant symmetric positive-definite matrix for function approximation errors (see https://arxiv.org/abs/1903.08894).
    You can also add errors into the Bellman optimal operator :math:`\\mathcal{T}^{*}` by "noise_scale".
    If :math:`D_\\rho` and :math:`K` are identity matrix and "noise_scale" is 0, the above equation becomes the ideal update.

    Options:
        noise_scale:
            Scale of noise in :math:`\\mathcal{T}^{*}` for error tolerance analysis.

        use_oracle_visitation:
            Set an identity matrix to :math:`D_{\\rho}` if True.

        num_samples:
            Number of samples to create :math:`D_{\\rho}`. We assign a state and action pair to each sample and set the number of occurrence as the probability mass function. The update becomes contraction if all the diagonal elements are positive and thus more samples may prevent deadly triad (see https://arxiv.org/abs/1903.08894).

        no_approx:
            Set an identity matrix to :math:`K` if True.

        diag_scale:
            Controls the magnitude of the diagonal elements of :math:`K` w.r.t. non-diagonal ones. If the non-diagonal elements are huge w.r.t. diagonal elements, contraction is not guaranteed (see https://arxiv.org/abs/1903.08894).
    """

    def run(self, num_steps=10000):
        assert self.is_tabular
        for _ in tqdm(range(num_steps)):
            self.record_history()
            new_values = self.compute_new_values()
            self.update(new_values)
            self.step += 1

    def update(self, new_values):
        values, new_values = self.tb_values.reshape(-1), new_values.reshape(-1)

        # ----- add noise into the optimal operator -----
        noise_scale = self.solve_options["noise_scale"]
        new_values = new_values + np.random.randn(*new_values.shape) * noise_scale

        # ----- make visitation matrix -----
        # for memory efficiency, we use elementwise multiplication instead of inner product.
        if self.solve_options["use_oracle_visitation"]:
            visitation = np.ones_like(values)  # SA
        else:
            # Randomly select states and action pairs for the number of "num_samples".
            num_samples = self.solve_options["num_samples"]
            index, counts = np.unique(
                np.random.randint(len(values), size=num_samples), return_counts=True
            )
            visitation = np.zeros_like(values)  # SA
            visitation[index] = counts / num_samples

        # ----- make approximation matrix -----
        if self.solve_options["no_approx"]:
            values = values + self.solve_options["lr"] * visitation * (
                new_values - values
            )
        else:
            SA = len(new_values)
            approx = np.random.rand(SA, SA)
            approx = approx @ approx.T
            scale = 1 / self.solve_options["diag_scale"]
            diag = np.ones((SA, SA)) * scale + np.eye(SA) * (1 - scale)
            approx = (approx / approx.max()) * diag
            values = values + self.solve_options["lr"] * approx @ (
                visitation * (new_values - values)
            )
        self.record_array("Values", values.reshape(self.dS, self.dA))
        self.set_tb_policy()

    def record_history(self):
        if self.step % self.solve_options["evaluation_interval"] == 0:
            expected_return = self.env.compute_expected_return(self.tb_policy)
            self.record_scalar("Return", expected_return, tag="Policy")


class OracleViSolver(OracleSolver):
    def compute_new_values(self):
        q = self.tb_values
        discount = self.solve_options["discount"]
        v = np.max(q, axis=-1)  # S
        new_q = self.env.reward_matrix + discount * (
            self.env.transition_matrix * v
        ).reshape(self.dS, self.dA)
        return np.asarray(new_q)

    def set_tb_policy(self):
        # set greedy policy
        policy = utils.eps_greedy_policy(self.tb_values, eps_greedy=0.0)
        self.record_array("Policy", policy)


class OracleCviSolver(OracleSolver):
    """
    Implementation of Conservative Value Iteration (CVI).
    See http://proceedings.mlr.press/v89/kozuno19a.html.
    """

    def compute_new_values(self):
        discount = self.solve_options["discount"]
        er_coef, kl_coef = self.solve_options["er_coef"], self.solve_options["kl_coef"]
        alpha, beta = kl_coef / (er_coef + kl_coef), 1 / (er_coef + kl_coef)
        p = self.tb_values

        # update preference
        mP = self.max_operator(p, beta)
        new_p = (
            alpha * (p - mP.reshape(-1, 1))
            + self.env.reward_matrix
            + discount * (self.env.transition_matrix * mP).reshape(self.dS, self.dA)
        )
        return np.asarray(new_p)

    def set_tb_policy(self):
        er_coef, kl_coef = self.solve_options["er_coef"], self.solve_options["kl_coef"]
        beta = 1 / (er_coef + kl_coef)
        policy = utils.softmax_policy(self.tb_values, beta=beta)
        self.record_array("Policy", policy)


class OracleMviSolver(OracleSolver):
    """
    Implementation of Munchausen Value Iteration (MVI).
    MVI is equivalent to CVI.
    See https://arxiv.org/abs/2007.14430.
    """

    def compute_new_values(self):
        discount = self.solve_options["discount"]
        er_coef, kl_coef = self.solve_options["er_coef"], self.solve_options["kl_coef"]
        # alpha controls the ratio of kl and entropy coefficients
        # tau is the base coefficient.
        alpha = kl_coef / (kl_coef + er_coef)
        tau = kl_coef + er_coef

        # update q
        policy = utils.softmax_policy(self.tb_values, beta=1 / tau)
        q = self.tb_values
        next_max = np.sum(policy * (q - tau * np.log(policy)), axis=-1)
        new_q = (
            self.env.reward_matrix
            + alpha * tau * np.log(policy)
            + discount
            * (self.env.transition_matrix * next_max).reshape(self.dS, self.dA)
        )
        return np.asarray(new_q)

    def set_tb_policy(self):
        er_coef, kl_coef = self.solve_options["er_coef"], self.solve_options["kl_coef"]
        tau = kl_coef + er_coef
        policy = utils.softmax_policy(self.tb_values, beta=1 / tau)
        self.record_array("Policy", policy)
