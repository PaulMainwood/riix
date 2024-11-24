"""base class for online rating systems"""
from abc import ABC
from typing import Optional
import numpy as np
from riix.utils import MatchupDataset


class OfflineRatingSystem(ABC):
    """
    Base class for offline rating systems. This class provides a framework for implementing
    various offline rating systems, WHR or Trueskill through time. The critical difference
    is that offline rating systems evaluate all games in the dataset, and iterate through 
    them each time - i.e., new games change past ratings. As a result, we need to import games
    and iterate/update the rating method separately. As a result, they do not have the distinction
    between "batched" and "online". Instead, you add all games in a rating period at once "add_games", and then
    iterate separately. 

    Attributes:
        rating_dim (int): Dimension of competitor ratings. This could be 1 for systems like Elo,
                          where each competitor has a single rating value, or more for systems
                          like TrueSkill that use multiple values (e.g., mean and standard deviation).
        competitors (list): A list of competitors within the rating system.
        num_competitors (int): The number of competitors in the system.
    """

    rating_dim: int

    def __init__(self, competitors):
        """
        Initializes a new instance of an online rating system with a list of competitors.

        Parameters:
            competitors (list): A list of competitors to be included in the rating system. Each
                                competitor should have a structure or identifier compatible with
                                the specific rating system's requirements.
        """
        self.competitors = competitors
        self.num_competitors = len(competitors)

    def print_leaderboard(self, num_places=None):
        """
        Prints the leaderboard of the rating system.

        Parameters:
            num_places int: The number of top places to display on the leaderboard.
        """
        pass  # Implementation should be provided by subclasses.

    def predict(
        self,
        matchups: np.ndarray,
        time_step: int = None,
        set_cache: bool = False,
    ):
        raise NotImplementedError

    def add_games(self, matchups: np.ndarray, outcomes: np.ndarray, time_step: int, use_cache=False, **kwargs):
        """
        Performs a batched addition based on a series of matchups and their outcomes with their associated time-steps.

        Parameters:
            matchups (np.ndarray): Array of matchups, where each matchup is represented by a pair of player indices
            outcomes (np.ndarray): Array of outcomes corresponding to each matchup represented as win (1), loss (0), or draw (0.5).
            time_step (int): The current time step or period of the rating update, used to adjust ratings over time.
            use_cache (bool, optional): Whether to use values cached during a prior call to predict() to speed up calculations. Defaults to False.
        """
        raise NotImplementedError

    def get_pre_match_ratings(self, matchups: np.ndarray, time_step: Optional[int]) -> np.ndarray:
        """
        Returns the ratings for competitors at the timestep of the matchups
        Useful when using pre-match ratings as features in downstream ML pipelines

        Parameters:
            matchups (np.ndarray of shape (n,2)): competitor indices
            time_step (optional int)

        Returns:
            np.ndarray of shape (n,2): ratings for specified competitors
        """
        raise NotImplementedError

    def iterate(self):
        raise NotImplementedError

    def fit_dataset(
        self,
        dataset: MatchupDataset,
        return_pre_match_probs: bool = False,
        return_pre_match_ratings: bool = False,
        n_iterations_per_time_period = 1,
        cache: bool = False,
    ):
        """Evaluate a rating system on a dataset. If you use this without at least one of return_pre_match_probs
        or return_pre_match_ratings set to true, this becomes an extra-slow version of adding games and then
        iterating for an offline algorithm"""

        n_matchups = len(dataset)
        if return_pre_match_probs:
            pre_match_probs = np.empty(shape=(n_matchups))
        if return_pre_match_ratings:
            pre_match_ratings = np.empty(shape=(n_matchups, 2 * self.rating_dim))

        idx = 0
        for matchups, outcomes, time_step in dataset:
            if return_pre_match_probs:
                batch_probs = self.predict(matchups = matchups, time_step = time_step, set_cache = cache)
                pre_match_probs[idx : idx + batch_probs.shape[0]] = batch_probs
            if return_pre_match_ratings:
                raise NotImplementedError
            
            self.iterate(n_iterations_per_time_period)
            idx += matchups.shape[0]

        if return_pre_match_probs and return_pre_match_ratings:
            return pre_match_probs, pre_match_ratings
        elif return_pre_match_probs:
            return pre_match_probs
        elif return_pre_match_ratings:
            return_pre_match_ratings
