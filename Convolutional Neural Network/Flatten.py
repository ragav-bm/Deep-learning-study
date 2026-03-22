import numpy as np

class Flatten:
    def __init__(self):
        self.trainable = False

    def forward(self, input_tensor):
        self.inp_original_shape = np.shape(input_tensor)
        return np.reshape(input_tensor,(self.inp_original_shape[0],-1))

    def backward(self, error_tensor):
        return np.reshape(error_tensor,self.inp_original_shape)
