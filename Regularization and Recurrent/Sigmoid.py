import numpy as np

from Layers.Base import BaseLayer
class Sigmoid(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        out = 1/(1+np.exp(-1*input_tensor))
        self.output = out
        return self.output

    def backward(self, error_tensor):
        out = error_tensor * self.output *(1- self.output)
        return out
