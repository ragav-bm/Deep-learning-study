import numpy as np

class Constant:
    def __init__(self, const_val):
        self.const_val = const_val
    def initialize(self, weights_shape, fan_in, fan_out):
        wght_tensor = np.zeros(weights_shape) + self.const_val
        return wght_tensor

class UniformRandom:
    def initialize(self, weights_shape, fan_in, fan_out):
        return np.random.uniform(0,1, weights_shape)

class Xavier:
    def initialize(self, weights_shape, fan_in, fan_out):
        sigma_ = np.sqrt(2/(fan_in+fan_out))
        return np.random.normal(0,sigma_, weights_shape)

class He:
    def initialize(self, weights_shape, fan_in, fan_out):
        sigma_ = np.sqrt(2/fan_in)
        return np.random.normal(0,sigma_,weights_shape)




