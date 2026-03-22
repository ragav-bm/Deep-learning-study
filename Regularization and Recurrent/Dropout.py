import numpy as np
from Layers.Base import BaseLayer

class Dropout(BaseLayer):
    def __init__(self,probability):
        self.probability = probability
        super().__init__()
        self.masking = None

    
    def forward(self, input_tensor):
        if self.testing_phase:
            return input_tensor
        
        else:
            self.masking = np.random.choice([1, 0], size=input_tensor.shape, p=[self.probability, 1 - self.probability])
            return input_tensor * (self.masking /(self.probability))

    def backward(self, error_tensor):
        if self.testing_phase:
            return error_tensor
        
        else:
            return error_tensor * (self.masking /(self.probability))