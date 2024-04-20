"""
Glicko 2
paper: http://www.glicko.net/research/dpcmsv.pdf
example: http://www.glicko.net/glicko/glicko2.pdf

"""
import math
import numpy as np
from riix.core.base import OnlineRatingSystem
from riix.utils.math_utils import sigmoid, sigmoid_scalar
from riix.utils.constants import THREE_OVER_PI_SQUARED


class Glicko2(OnlineRatingSystem):
    """
    Implements the Glicko 2 rating system, designed by Mark Glickman.
    """

    rating_dim = 2

    def __init__(
        self,
        competitors: list,
        initial_rating: float = 1500.0,
        initial_rd: float = 350.0,
        initial_sigma: float = 0.06,
        tau: float = 0.5,
        epsilon: float = 1e-6,
        dtype=np.float64,
        update_method='iterative',
    ):
        """Initializes the Glicko rating system with the given parameters."""
        super().__init__(competitors)
        self.mus = np.zeros(shape=self.num_competitors, dtype=dtype) + ((initial_rating - 1500.0) / 173.7178)
        self.initial_phi = initial_rd / 173.7178
        self.phis = np.zeros(shape=self.num_competitors, dtype=dtype) + self.initial_phi
        self.sigmas = np.zeros(shape=self.num_competitors, dtype=dtype) + initial_sigma
        self.has_played = np.zeros(shape=self.num_competitors, dtype=np.bool_)
        self.tau = tau
        self.tau2 = tau**2.0
        self.epsilon = epsilon

        if update_method == 'batched':
            self.update = self.batched_update
        elif update_method == 'iterative':
            self.update = self.iterative_update

    @staticmethod
    def g_scalar(phi):
        """this is DIFFERENT from g in regular Glicko"""
        return 1.0 / math.sqrt(1.0 + (THREE_OVER_PI_SQUARED * (phi**2.0)))

    @staticmethod
    def g_vector(phi):
        """vector version"""
        return 1.0 / np.sqrt(1.0 + (THREE_OVER_PI_SQUARED * np.square(phi)))

    # TODO should Glicko probs incorporate the dev increase?
    def predict(self, matchups: np.ndarray, time_step: int = None, set_cache: bool = False):
        """generate predictions"""
        mu_1 = self.mus[matchups[:, 0]]
        mu_2 = self.mus[matchups[:, 1]]
        mu_diff = mu_1 - mu_2
        phi_1 = self.phis[matchups[:, 0]]
        phi_2 = self.phis[matchups[:, 1]]
        # the papers and theory seem to indicate this...
        # combined_phi = self.g_vector(np.sqrt(np.square(phi_1) + np.square(phi_2)))
        # but this seems to work better...
        combined_phi = self.g_vector(phi_1 + phi_2)
        probs = sigmoid(combined_phi * mu_diff)
        return probs

    def get_pre_match_ratings(self, matchups: np.ndarray, **kwargs):
        means = self.mus[matchups]
        devs = self.phis[matchups]
        ratings = np.concatenate((means[..., None], devs[..., None]), axis=2).reshape(means.shape[0], -1)
        return ratings

    def increase_rating_dev(self, matchups):
        """called once per period to model the increase in variance over time"""
        active_in_period = np.unique(matchups)
        self.has_played[active_in_period] = True
        self.phis[self.has_played] = np.minimum(
            np.sqrt(np.square(self.phis[self.has_played]) + np.square(self.sigmas[self.has_played])), self.initial_phi
        )
        return active_in_period

    def batched_update(self, matchups, outcomes, use_cache=False):
        """apply one update based on all of the results of the rating period"""
        pass
        # active_in_period = np.unique(matchups) # (n_active,)
        # active_mask = np.equal(matchups[:, :, None], active_in_period[None, :])  # (M,2,n_active)
        # mus = self.mus[matchups]
        # phis = self.phis[matchups]
        # sigmas = self.sigmas[matchups]
        # gs = self.g_vector(phis)
        # mu_diffs = mus[:,0] - mus[:,1]
        # probs = sigmoid(gs * mu_diffs) # (M,2) where col 0 is prob_2 and col 1 is prob_1

    def f(self, x, delta2, phi2, v, a):
        ex = math.exp(x)
        phi2_v_ex = phi2 + v + ex
        num_1 = ex * (delta2 - phi2_v_ex)
        denom_1 = 2 * ((phi2_v_ex) ** 2.0)
        term_2 = (x - a) / self.tau2
        return (num_1 / denom_1) - term_2

    def get_sigma_star(self, phi, delta, v, sigma):
        delta2 = delta**2.0
        phi2 = phi**2.0
        A = a = math.log(sigma**2.0)
        if delta2 > (phi2 + v):
            B = math.log(delta2 - phi2 - v)
        else:
            B = a - self.tau
            while self.f(B, delta2=delta2, phi2=phi2, v=v, a=a) < 0:
                B -= self.tau

        f_A = self.f(A, delta2, phi2, v, a)
        f_B = self.f(B, delta2, phi2, v, a)
        while math.fabs(A - B) > self.epsilon:
            C = A + ((A - B) * f_A) / (f_B - f_A)
            f_C = self.f(C, delta2, phi2, v, a)
            if (f_C * f_B) <= 0:
                A = B
                f_A = f_B
            else:
                f_A = f_A / 2.0
            B = C
            f_B = f_C
        sigma_prime = math.exp(A / 2.0)
        return sigma_prime

    def iterative_update(self, matchups, outcomes, **kwargs):
        """treat the matchups in the rating period as if they were sequential"""
        self.increase_rating_dev(matchups)
        for idx in range(matchups.shape[0]):
            comp_1, comp_2 = matchups[idx]
            mu_1 = self.mus[comp_1]
            mu_2 = self.mus[comp_2]
            mu_diff = mu_1 - mu_2
            phi_1 = self.phis[comp_1]
            phi_2 = self.phis[comp_2]
            g_1 = self.g_scalar(phi_1)
            g_2 = self.g_scalar(phi_2)
            p_1 = sigmoid_scalar(g_2 * mu_diff)
            p_2 = sigmoid_scalar(-g_1 * mu_diff)
            v_1 = 1.0 / ((g_2**2.0) * p_1 * (1.0 - p_1))
            v_2 = 1.0 / ((g_1**2.0) * p_2 * (1.0 - p_2))

            delta_1 = v_1 * g_2 * (outcomes[idx] - p_1)
            delta_2 = v_2 * g_1 * (1.0 - outcomes[idx] - p_2)

            sigma_star_1 = self.get_sigma_star(phi_1, delta_1, v_1, self.sigmas[comp_1])
            sigma_star_2 = self.get_sigma_star(phi_2, delta_2, v_2, self.sigmas[comp_2])

            self.sigmas[comp_1] = sigma_star_1
            self.sigmas[comp_2] = sigma_star_2

            # I guess don't do this since I update them all at the beginning?
            phi_star_1 = (self.phis[comp_1] ** 2.0) + (sigma_star_1**2.0)
            phi_star_2 = (self.phis[comp_2] ** 2.0) + (sigma_star_2**2.0)

            self.phis[comp_1] = 1.0 / math.sqrt((1.0 / phi_star_1) + (1.0 / (v_1**2.0)))
            self.phis[comp_2] = 1.0 / math.sqrt((1.0 / phi_star_2) + (1.0 / (v_2**2.0)))

            self.mus[comp_1] += (self.phis[comp_1] ** 2.0) * g_1 * (outcomes[idx] - p_1)
            self.mus[comp_2] += (self.phis[comp_2] ** 2.0) * g_2 * (1.0 - outcomes[idx] - p_2)

    def print_leaderboard(self, num_places):
        sort_array = self.ratings - (3.0 * self.rating_devs)
        sorted_idxs = np.argsort(-sort_array)[:num_places]
        max_len = min(np.max([len(comp) for comp in self.competitors] + [10]), 25)
        print(f'{"competitor": <{max_len}}\t{"rating - (3*dev)"}\t')
        for p_idx in range(num_places):
            comp_idx = sorted_idxs[p_idx]
            print(f'{self.competitors[comp_idx]: <{max_len}}\t{sort_array[comp_idx]:.6f}')
