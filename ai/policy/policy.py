
class Policy:

    def evaluate(self, features):
        raise NotImplementedError()

    def get_feature_vector(self, board):
        raise NotImplementedError()

    def add_reward(self, reward):
        raise NotImplementedError()

    def choose(self, options):
        raise NotImplementedError()