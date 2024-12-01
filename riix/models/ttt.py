"""a template to copy and paste when implementing a new rating system"""
import numpy as np
from riix.core.base_offline import OfflineRatingSystem
import trueskill_through_time as ttt

class TrueSkillThroughTime(OfflineRatingSystem):
    """
    Trueskill through time, an offline rating system building on the Trueskill
    online rating system, but sweeping the update back and forward rather than just
    a forward pass. Paper here: https://www.herbrich.me/papers/ttt.pdf.
    """

    rating_dim = 2

    def __init__(
        self,
        dataset, 
        beta = 1.0,
        mu = 0.0,
        p_draw = 0.0,
        epsilon = 0.01,
        iterations = 30
    ):
        """
        Initializes the TTT system with the given parameters.

        Parameters:
            BETA: The beta variable, which controls the rate at which ratings change. Defaults to 1.0
            update_method (str, optional): Method used to update ratings ('online' or other methods if implemented). Defaults to 'batched' and at present 'online' is not implemented
            dtype: The data type for internal numpy computations. Defaults to np.float64.

        Initializes an WHR rating system with customizable settings for w2 value (the "twitchiness" of the rating).
        """
        self.beta = beta
        self.mu = mu
        self.p_draw = p_draw
        self.epsilon = epsilon
        self.iterations = iterations
        
        a = TrueSkillThroughTime._riix_to_ttt(dataset)
        self.ttt_history = ttt.History(composition = a[0], times = a[1], sigma = 1.6, gamma = 0.036)


    def predict(self, matchups: np.ndarray, time_step: int = None, set_cache: bool = False):
        """
        Generates a numpy array for predictions for a series of matchups between competitors in riix format.
        """
        probabilities = np.zeros(len(matchups))
    
        for i, (player1, player2) in enumerate(matchups):
            # Convert IDs to strings since the function expects strings
            p1_prob, _ = self.whr.probability_future_match(str(player1), str(player2))
            # Store only the first player's probability
            probabilities[i] = p1_prob
        
        return probabilities

    def get_pre_match_ratings(self, matchups: np.ndarray, **kwargs):
        return self.ratings[matchups]

    def add_games(self, dataset, use_cache=False, **kwargs):
        """
        Performs a batched addition based on a series of matchups and their outcomes with their associated time-steps.

        Parameters:
            matchups (np.ndarray): Array of matchups, where each matchup is represented by a pair of player indices
            outcomes (np.ndarray): Array of outcomes corresponding to each matchup represented as win (1), loss (0), or draw (0.5).
            time_step (int): The current time step or period of the rating update, used to adjust ratings over time.
            use_cache (bool, optional): Whether to use values cached during a prior call to predict() to speed up calculations. Defaults to False.
        """
        fixtures, timestamps = TrueSkillThroughTime._riix_to_ttt(dataset)
        self.ttt_history.add_games(fixtures, [], timestamps, [])

    def iterate(self, n = 10):
        self.ttt_history.convergence(epsilon = self.epsilon, iterations = n)

    @staticmethod
    def _riix_to_ttt(dataset):
        """
        Converts a riix dataset of games to the format used by the ttt package
        """
        matchups = [[[x], [y]] for x, y in dataset.matchups]
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
            self.ttt_history.add_games(fixtures, [], timestamps, [])
            self.iterate(iterations)
            return None
            
        # For pre-match probabilities, we need to process day by day
        probs = []
        unique_times = sorted(set(timestamps))
        
        for time in unique_times:
            # Find matches for this timestamp
            day_indices = [i for i, t in enumerate(timestamps) if t == time]
            day_fixtures = [fixtures[i] for i in day_indices]
            day_timestamps = [timestamps[i] for i in day_indices]
            
            # Get predictions for this day's matches using current ratings
            for fixture in day_fixtures:
                player1, player2 = fixture[0][0], fixture[1][0]  # Extract player IDs
                # Get probability for first player winning
                p1_prob = self.predict_matchup(player1, player2, time)
                probs.append(p1_prob)
                
            # Add this day's matches and update ratings
            self.ttt_history.add_games(day_fixtures, [], day_timestamps, [])
            self.iterate(iterations)
        
        return np.array(probs)
