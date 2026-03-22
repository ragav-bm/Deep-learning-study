import numpy as np
import math
from Layers.Base import BaseLayer

class TanH(BaseLayer):
    def __init__(self):
        self.activations = None
        super().__init__()


    
    def forward(self, input_tensor):
        self.activations = np.tanh(input_tensor)
        return self.activations

    def backward(self, error_tensor):
        return error_tensor * (1 - self.activations ** 2)