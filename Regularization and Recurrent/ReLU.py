from Layers.Base import BaseLayer
import numpy as np

class ReLU(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        self.input_tensor = [np.maximum(0, i) for i in input_tensor]
        return self.input_tensor

    def backward(self, error_tensor):
        self.derivativeReLU = np.array(self.input_tensor)

        self.derivativeReLU = np.where([i>0 for i in self.input_tensor],1,0)
        return self.derivativeReLU * error_tensor
