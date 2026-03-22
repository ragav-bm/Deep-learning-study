import numpy as np

class L2_Regularizer:
    def __init__(self, alpha):
        self.alpha = alpha

    def calculate_gradient(self, weights):
        subgrad = weights * self.alpha
        return subgrad

    def norm(self, weights):
        normloss = (np.linalg.norm(weights) ** 2) * self.alpha
        return normloss


class L1_Regularizer:
    def __init__(self, alpha):
        self.alpha = alpha

    def calculate_gradient(self, weights):
        subgrad = np.sign(weights)*self.alpha
        return subgrad

    def norm(self, weights):
        normloss = np.sum(np.abs(weights))*self.alpha
        return normloss
