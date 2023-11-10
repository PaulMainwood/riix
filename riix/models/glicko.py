"""Glicko"""
import math
import numpy as np
from riix.core.base import OnlineRatingSystem
from riix.utils.math_utils import sigmoid


Q = math.log(10.0) / 400.0
Q2 = Q**2.0
Q2_3 = 3.0 * Q2
PI2 = math.pi**2.0


def g(rating_dev):
    """the g function"""
    return 1.0 / np.sqrt(1.0 + (Q2_3 * np.square(rating_dev)) / PI2)


class Glicko(OnlineRatingSystem):
    """the og glicko rating system shoutout to Mark"""

    def __init__(
        self,
        num_competitors: int,
        initial_rating: float = 1500.0,
        initial_rating_dev: float = 350.0,
        c: float = 0.0,
        dtype=np.float64,
    ):
        self.num_competitors = num_competitors
        self.initial_rating_dev = initial_rating_dev
        self.c2 = c**2.0
        self.ratings = np.zeros(shape=num_competitors, dtype=dtype) + initial_rating
        self.rating_devs = np.zeros(shape=num_competitors, dtype=dtype) + initial_rating_dev
        self.has_played = np.zeros(shape=num_competitors, dtype=np.bool_)

    def predict(self, time_step: int, matchups: np.ndarray, set_cache: bool = False):
        """generate predictions"""
        ratings_1 = self.ratings[matchups[:, 0]]
        ratings_2 = self.ratings[matchups[:, 1]]
        rating_devs_1 = self.rating_devs[matchups[:, 0]]
        rating_devs_2 = self.rating_devs[matchups[:, 1]]
        combined_dev = g(np.sqrt(np.square(rating_devs_1) + np.square(rating_devs_2)))
        rating_diffs = ratings_1 - ratings_2
        probs = sigmoid(Q * combined_dev * rating_diffs)
        return probs

    def fit(
        self,
        time_step: int,
        matchups: np.ndarray,
        outcomes: np.ndarray,
        use_cache: bool = False,
        update_method: str = 'batched',
    ):
        if update_method == 'batched':
            self.batched_update(matchups, outcomes, use_cache)
        elif update_method == 'iterative':
            self.iterative_update(matchups, outcomes)

    def batched_update(self, matchups, outcomes, use_cache=False):
        """apply one update based on all of the results of the rating period"""
        active_in_period = np.unique(matchups)
        self.has_played[active_in_period] = True
        self.rating_devs[self.has_played] = np.minimum(
            np.sqrt(np.square(self.rating_devs[self.has_played]) + self.c2), self.initial_rating_dev
        )
        masks = np.equal(matchups[:, :, None], active_in_period[None, :])  # N x 2 x active

        ratings = self.ratings[matchups]
        rating_devs = self.rating_devs[matchups]
        rating_diffs = ratings[:, 0] - ratings[:, 1]
        probs_1 = sigmoid(Q * rating_devs[:, 1] * rating_diffs)
        probs_2 = sigmoid(-1.0 * (Q * rating_devs[:, 0] * rating_diffs))
        g_rating_devs = g(rating_devs)

        tmp = np.stack([probs_1 * (1 - probs_1), probs_2 * (1 - probs_2)]).T * np.square(g_rating_devs)[:, [1, 0]]
        d2 = 1 / ((tmp[:, :, None] * masks).sum(axis=(0, 1)) * Q2)

        outcomes = np.hstack([outcomes[:, None], 1.0 - outcomes[:, None]])
        probs = np.hstack([probs_1[:, None], probs_2[:, None]])

        r_num = Q * ((g_rating_devs[:, [1, 0]] * (outcomes - probs))[:, :, None] * masks).sum(axis=(0, 1))
        r_denom = (1 / np.square(self.rating_devs[active_in_period])) + (1 / d2)

        self.ratings[active_in_period] += r_num / r_denom
        self.rating_devs[active_in_period] = np.sqrt(1.0 / r_denom)

    def iterative_update(self, matchups, outcomes):
        """treat the matchups in the rating period as if they were sequential"""
        pass
        # for idx in range(matchups.shape[0]):
        #     comp_1, comp_2 = matchups[idx]
        #     diff = self.ratings[comp_1] - self.ratings[comp_2]
        #     prob = sigmoid(self.alpha * diff)
        #     update = self.k * (outcomes[idx] - prob)
        #     self.ratings[comp_1] += update
        #     self.ratings[comp_2] -= update
