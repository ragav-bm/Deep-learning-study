import numpy as np
from Layers.Base import BaseLayer

class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        expo = np.exp(input_tensor - np.max(input_tensor,axis=-1, keepdims=True))
        sum_expo = np.sum(expo, axis=-1, keepdims=True)
        class_probability = expo/sum_expo
        self.probability_distb = class_probability
        return self.probability_distb

    def backward(self, error_tensor):
        # print(self.probability_distb.shape)
        loss = error_tensor - np.sum(error_tensor * self.probability_distb, axis=1, keepdims=True)
        loss *= self.probability_distb
        return loss
