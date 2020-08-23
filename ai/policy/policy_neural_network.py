from torch.distributions.multinomial import Multinomial

from ai.lib.decider import Decider
from ai.policy.policy import Policy


class NeuralNetworkPolicy(Policy):

    def __init__(self, neural_network: Decider):
        self.net = neural_network

    def get_feature_vector(self, board):
        pass

    def argmax(self, move_ratings: [float]):
        return move_ratings.index(max(move_ratings))

    def evaluate(self, possible_boards):
        move_ratings = []
        for board in possible_boards:
            value_of_board = self.net.predict(self.get_feature_vector(board))
            move_ratings.append(value_of_board)

        # If randomization of move is needed for exploration
        # we do that here. Randomization % is called 'epsilon' in
        # the config file.
        move = 0
        self.net.run_decision(self.get_feature_vector(possible_boards[move]))

        return move

    def choose(self, options):
        distribution = Multinomial(1, probabilities)
        move = distribution.sample()
        self.saved_log_probabilities.append(distribution.log_prob(move))

        _, move = move.max(0)
        # calculate the value estimation and save for backward
        value_estimate =
        self.saved_value_estimations.append(value_estimate)
        return move

    def add_reward(self, reward):
        self.net.give_reward_to_nn(reward)
