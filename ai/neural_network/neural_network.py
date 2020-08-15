class NeuralNetwork:

    def give_reward(self, reward):
        raise NotImplementedError()

    def choose(self, options) -> int:
        raise NotImplementedError()
