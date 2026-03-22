import numpy as np


class Optimizer:
    def __init__(self):
        self.regularizer = None

    def add_regularizer(self, regularizer):
        self.regularizer = regularizer

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.regularizer:
            reg_grad = self.regularizer.calculate_gradient(weight_tensor)
            weight_tensor -= self.learning_rate * reg_grad

        return weight_tensor




class Sgd(Optimizer):
    def __init__(self, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        weight_tensor = super().calculate_update(weight_tensor,gradient_tensor)
        weight_tensor = weight_tensor - self.learning_rate * gradient_tensor
        return weight_tensor


class SgdWithMomentum(Optimizer):
    def __init__(self, learning_rate, momentum_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.momementum_dir = 0.

    def calculate_update(self, weight_tensor, gradient_tensor):
        momementum_dir = self.learning_rate * gradient_tensor + self.momentum_rate * self.momementum_dir
        weight_tensor = super().calculate_update(weight_tensor, gradient_tensor)
        weight_tensor = weight_tensor - momementum_dir
        self.momementum_dir = momementum_dir
        return weight_tensor

class Adam(Optimizer):
    def __init__(self, learning_rate, beta1, beta2):
        super().__init__()
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.m1 = 0.
        self.m2 = 0.
        self.timestep = 1

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.m1 = self.beta1 * self.m1 + (1 - self.beta1) * gradient_tensor
        self.m2 = self.beta2 * self.m2 + (1 - self.beta2) * np.power(gradient_tensor, 2)
        m1_hat = self.m1 / (1 - np.power(self.beta1, self.timestep))
        m2_hat = self.m2 / (1 - np.power(self.beta2, self.timestep))
        self.timestep += 1

        # Update weights
        weight_tensor = super().calculate_update(weight_tensor, gradient_tensor)
        weight_tensor = weight_tensor - self.learning_rate * (m1_hat / (np.sqrt(m2_hat) + np.finfo(float).eps))

        return weight_tensor

