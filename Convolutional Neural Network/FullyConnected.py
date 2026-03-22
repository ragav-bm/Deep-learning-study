from .Base import BaseLayer
import numpy as np


class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        super().__init__()
        self.trainable = True
        self.weights1 = np.random.uniform(0, 1.0, size=(self.input_size, self.output_size))

        self.bias = np.random.uniform(0, 1.0, size=(self.output_size))
        self.weights_bias = np.random.uniform(0, 1.0, size=(1, self.output_size))
        self.weights = np.append(self.weights1, self.weights_bias, axis=0)
        self._optimizer = None

    def forward(self, input_tensor):
        input_tensor = np.array(input_tensor)
        input_tensor_bias = np.ones((input_tensor.shape[0], 1))
        input_tensor = np.append(input_tensor, input_tensor_bias, axis=1)
        self.input_tensor = input_tensor
        # print("weights ")
        # print( self.weights.shape)
        # print("input_tensor ")
        # print( self.input_tensor.shape)
        output_tensor = np.dot(input_tensor, self.weights)
        return output_tensor

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    @property
    def gradient_weights(self):
        return self._gradient_weights

    def backward(self, error_tensor):
        gradient_error = np.dot(error_tensor, self.weights[:-1, :].T)
        self._gradient_weights = np.dot(self.input_tensor.T, error_tensor)

        self._gradient_bias = np.sum(error_tensor, axis=0)
        if self._optimizer:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
            self.bias = self._optimizer.calculate_update(self.bias, self._gradient_bias)

        return gradient_error

    def initialize(self, weights_initializer, bias_initializer):
        inp = self.input_size
        out = self.output_size
        self.weights[-1] = bias_initializer.initialize((1,self.output_size),inp,out)
        self.weights[:-1] = weights_initializer.initialize((self.input_size,self.output_size),inp,out)





