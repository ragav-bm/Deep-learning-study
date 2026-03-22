from Layers.Base import BaseLayer
from Layers.Helpers import compute_bn_gradients

import numpy as np

class BatchNormalization(BaseLayer):
    def __init__(self, channels):
        self.channels = channels
        super().__init__()
        self.trainable = True
        self.epsi = np.finfo(np.float64).eps
        self.decay = 0.8
        self.running_mean = None
        self.running_variance = None
        # self.running_mean = np.zeros(self.channels, dtype=np.float32)
        # self.running_variance = np.ones(self.channels, dtype=np.float32)
        self.optimizer = None
        self.initialize(None, None)

    def initialize(self, weights_init, bias_init):
        self.weights = np.ones((1,self.channels))
        self.bias = np.zeros((1,self.channels))


    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.reform_inp = input_tensor
        if len(input_tensor.shape) == 4:
            input_tensor = self.reformat(input_tensor)
            self.reform_inp = input_tensor
        if self.testing_phase:
            # testing phase -> online estimatn
            mean = self.running_mean
            variance = self.running_variance
        if not self.testing_phase:
            # normal training phase code
            self.mean = np.mean(input_tensor, axis=0)
            self.variance = np.var(input_tensor, axis=0)
            if self.running_mean is None:
                self.running_mean = self.mean
                self.running_variance = self.variance
            else:
                self.running_mean = self.mean*self.decay + (1 - self.decay) * self.running_mean
                self.running_variance = self.variance*self.decay + (1 - self.decay) * self.running_variance

        self.norm_tensor = (input_tensor - self.mean) / np.sqrt(self.variance + self.epsi)
        yhat = self.weights*self.norm_tensor + self.bias
        if len(self.input_tensor.shape) == 4:
            yhat = self.reformat(yhat)
        return yhat

###############################
###############################

    def backward(self, error_tensor):
        reformatted_err = error_tensor
        if len(error_tensor.shape) == 4:
            reformatted_err = self.reformat(error_tensor)
            grad_inp = self.reformat(compute_bn_gradients(reformatted_err, self.reform_inp, self.weights, self.mean, self.variance))
        else:
            grad_inp = compute_bn_gradients(reformatted_err, self.reform_inp, self.weights, self.mean, self.variance)
        self.gradient_weights = np.sum(
            reformatted_err * self.norm_tensor, axis=0
        ).reshape(1, self.channels)
        self.gradient_bias = np.sum(reformatted_err, axis=0).reshape(1, self.channels)
        if self.optimizer is not None:
            self.bias = self.optimizer.calculate_update(self.bias, self.gradient_bias)
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
        return grad_inp


    def reformat(self, tensor):
        if len(tensor.shape) == 4:  #conv layer
            batch, ch, height, width = np.shape(tensor)
            return tensor.transpose(0, 2, 3, 1).reshape(batch * height * width, ch)

        if len(tensor.shape) == 2:  #fc
            spatial_sz, ch = tensor.shape
            self.inp_shape = np.shape(self.input_tensor)
            batch = self.inp_shape[0]
            height = self.inp_shape[2]
            width = self.inp_shape[3]
            return tensor.reshape(batch, height, width, ch).transpose(0, 3, 1, 2)
        else:
            pass

