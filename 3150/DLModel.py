import numpy as np

class DLModel:
    def __init__(self, name="Model"):
        self.layers = []
        self._is_compiled = False

    def add(self, layer):
        self.layers.append(layer)

    def compile(self):
        pass

    def compute_cost(self):
        pass

    def train(self):
        pass

    def predict(self):
        pass

    def squared_means(self, AL, Y):
        return (AL ** 2) - (Y ** 2)

    def squared_means_backward(self, AL, Y):
        pass

    def cross_entropy(self, AL, Y):
        error = np.where(Y == 0, -np.log(1 - AL), -np.log(AL))
        return error

    def cross_entropy_backward(self, AL, Y):
        dAL = np.where(Y == 0, 1 / (1 - AL), -1 / AL)
        return dAL

    def compile(self, loss, threshold=0.5):
        self.loss = loss

        if loss == "squared_means":
            self.loss_forward = self.squared_means
            self.loss_backward = self.squared_means_backward
        elif loss == "cross_entropy":
            self.loss_forward = self.cross_entropy
            self.loss_backward = self.cross_entropy_backward

        self.threshold = threshold
        self._is_compiled = True