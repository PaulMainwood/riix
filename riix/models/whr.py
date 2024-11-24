"""a template to copy and paste when implementing a new rating system"""
import numpy as np
from whr import whole_history_rating
from riix.core.base_offline import OfflineRatingSystem


class WHR(OfflineRatingSystem):
    """The Whole History Rating system designed by Rémi Coulom
    and described here. https://www.remi-coulom.fr/WHR/WHR.pdf.
    This implementation is just a wrapper around the implementation
    by Pierre-Francois Monville. 
    https://github.com/pfmonville/whole_history_rating
    The intent is to update this to a full implementation utilising JAX
    and other speed-ups.
    """

    rating_dim = 2

    def __init__(
        self,
        w2: float = 300.0,
        dtype = np.float64,
    ):
        """
        Initializes the WHR system with the given parameters.

        Parameters:
            competitors (list): A list of competitors to be rated within the system.
            w2 (float, optional): The w2, which controls the rate at which ratings change. Defaults to 300.0
            update_method (str, optional): Method used to update ratings ('online' or other methods if implemented). Defaults to 'batched' and at present 'online' is not implemented
            dtype: The data type for internal numpy computations. Defaults to np.float64.

        Initializes an WHR rating system with customizable settings for w2 value (the "twitchiness" of the rating).
        """
        self.w2 = w2
        
        self.whr = whole_history_rating.Base({w2: self.w2})


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
        games = WHR._riix_to_whr(dataset)
        self.whr.load_games(games)

    def iterate(self, n):
        self.whr.iterate(n)

    @staticmethod
    def _riix_to_whr(dataset):
        """
        Converts a riix dataset of games to the format used by the WHR package
        """
        formatted_list = []
    
        for (player1, player2), outcome, time_step in zip(dataset.matchups, dataset.outcomes, dataset.time_steps):
            # Use 'B' for 1.0 outcome and 'W' for 0.0 outcome
            result_char = 'B' if outcome == 1.0 else 'W'
            formatted_string = f"{player1} {player2} {result_char} {time_step}"
            formatted_list.append(formatted_string)
    
        return formatted_list

    
    def print_leaderboard(self, num_places):
        raise NotImplementedError
