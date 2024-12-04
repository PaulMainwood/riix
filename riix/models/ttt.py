"A wrapper aroud trueskill through time."

import numpy as np
from riix.core.base_offline import OfflineRatingSystem
import trueskill_through_time as ttt
import polars as pl

class TrueSkillThroughTime(OfflineRatingSystem):
    """
    Trueskill through time, an offline rating system building on the Trueskill
    online rating system, but sweeping the update back and forward rather than just
    a forward pass. Paper here: https://www.herbrich.me/papers/ttt.pdf.
    """

    rating_dim = 2

    def __init__(
        self,
        beta = 1.0,
        mu = 0.0,
        p_draw = 0.0,
        epsilon = 0.01,
    ):
        """
        Initializes an empty TTT system with the given parameters.

        Parameters:
            BETA: The beta variable, which controls the rate at which ratings change. Defaults to 1.0
            update_method (str, optional): Method used to update ratings ('online' or other methods if implemented). Defaults to 'batched' and at present 'online' is not implemented
            dtype: The data type for internal numpy computations. Defaults to np.float64.

        """
        self.beta = beta
        self.mu = mu
        self.p_draw = p_draw
        self.epsilon = epsilon
        
        self.ttt_history = ttt.History(composition = [], times = np.array([]), beta = beta)

    def predict(self, matchups: np.ndarray, time_step: int = None, set_cache: bool = False):
        return ttt.predict(self.ttt_history, 
                        matchups[:, 0],  # All player1s
                        matchups[:, 1],  # All player2s 
                        np.full(len(matchups), time_step))  # Array of times

    def get_pre_match_ratings(self, matchups: np.ndarray, **kwargs):
        return self.ratings[matchups]

    def add_games(self, dataset, use_cache=False, **kwargs):
        """
        Adds new games to the existing TTT model.
        
        Parameters:
            dataset: riix format dataset containing new matchups and timestamps
            use_cache: unused, kept for compatibility with base class
        """
        fixtures, timestamps = TrueSkillThroughTime._riix_to_ttt(dataset)
        self.ttt_history.add_games(fixtures, [], timestamps)

    def iterate(self, n = 10):
        self.ttt_history.convergence(epsilon = self.epsilon, iterations = n)

    @staticmethod
    def _riix_to_ttt(dataset):
        """
        Converts a riix dataset of games to the format used by the ttt package.
        Ensures player IDs are regular Python integers, not numpy integers.
        """
        matchups = [[[int(x)], [int(y)]] for x, y in dataset.matchups]
        timestamps = dataset.time_steps.tolist()

        return (matchups, timestamps)

    
    def print_leaderboard(self, num_places):
        raise NotImplementedError
    
    def fit_dataset(self, dataset, return_pre_match_probs = False, iterations = 5):
        """
        Fits the TTT model to a dataset, with optional pre-match probability calculation.
        
        When return_pre_match_probs is True, this method:
        1. Groups matches by day
        2. For each day:
            - Predicts outcomes for that day's matches using current ratings
            - Adds those matches to the history
            - Runs iterations to update ratings
        
        Parameters:
            dataset: A riix format dataset containing matchups and timestamps
            return_pre_match_probs: If True, calculate match probabilities before updating
            iterations: Number of iterations to run after adding each batch
            
        Returns:
            If return_pre_match_probs is True, returns array of pre-match probabilities
            Otherwise returns None
        """
        fixtures, timestamps = self._riix_to_ttt(dataset)
        
        if not return_pre_match_probs:
            # Simplest case - just add all games and iterate
            self.ttt_history.add_games(fixtures, times = timestamps)
            self.iterate(iterations)
            return None
            
        # For pre-match probabilities, process day by day
        probs = []
        unique_times = sorted(set(timestamps))
        
        for time in unique_times:
            # Find matches for this timestamp
            day_indices = [i for i, t in enumerate(timestamps) if t == time]
            day_fixtures = [fixtures[i] for i in day_indices]
            day_timestamps = [timestamps[i] for i in day_indices]
            
            # Convert day's fixtures to matchups format for prediction
            day_matchups = np.array([[fixture[0][0], fixture[1][0]] for fixture in day_fixtures])
            
            # Get predictions for all of this day's matches at once
            day_probs = self.predict(day_matchups, time_step=time)
            probs.extend(day_probs)
            
            # Add this day's matches and update ratings
            self.ttt_history.add_games(day_fixtures, times = day_timestamps)
            #The idea to iterate just on the matches doens't seem good.
            #self.ttt_history.iterate_on_matches(day_fixtures, day_timestamps, iterations = 5)
            self.iterate(iterations)
    
        return np.array(probs)