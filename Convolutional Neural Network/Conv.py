import numpy as np
from scipy.signal import correlate, correlate2d, convolve
from .Base import BaseLayer
import math
import copy

class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels

        # trainable
        super().__init__()
        self.trainable = True

        if len(self.convolution_shape) == 3:
            self.weights = np.random.uniform(0, 1.0, size=(
            self.num_kernels, self.convolution_shape[0], self.convolution_shape[1], self.convolution_shape[2]))
            self.bias = np.random.uniform(0, 1.0, size=(self.num_kernels,))
        else:
            self.weights = np.random.uniform(0, 1.0, size=(
            self.num_kernels, self.convolution_shape[0], self.convolution_shape[1]))
            self.bias = np.random.uniform(0, 1.0, size=(self.num_kernels,))

        # self.weights = None
        # self.bias = None
        self.g_weights = None
        self.g_bias = None
        self._optimizer = None

    def forward(self, input_tensor):
        # self.input_tensor = input_tensor

        if (len(input_tensor.shape) == 3):
            padding = (self.convolution_shape[1] - 1) / 2
            if padding % 2 == 0:
                p_1 = int(padding)
                p_2 = int(padding)

            else:
                p_1 = math.ceil(padding)
                p_2 = math.floor(padding)

            b, c, len1 = input_tensor.shape

            output_len = int((len1 + p_1 + p_2 - self.convolution_shape[1]) / self.stride_shape[0]) + 1
            output_tensor = np.zeros((b, self.num_kernels, output_len))

            input_padded = np.pad(input_tensor, pad_width=((0, 0), (0, 0), (p_2, p_1)), mode='constant',
                                  constant_values=0)

            for batch in range(b):
                for kernel in range(self.num_kernels):
                    for channel in range(c):
                        output_tensor[batch, kernel] = output_tensor[batch, kernel] + correlate(
                            input_padded[batch, channel], self.weights[kernel, channel], mode='valid')[
                                                                                      ::self.stride_shape[0]]

                    output_tensor[batch, kernel] = output_tensor[batch, kernel] + self.bias[kernel]


        else:
            b, c, height, width = input_tensor.shape
            padding_h = ((self.convolution_shape[1] - 1) / 2)
            padding_w = ((self.convolution_shape[2] - 1) / 2)
            if padding_h % 2 == 0:
                p_h_1 = int(padding_h)
                p_h_2 = int(padding_h)
            else:
                p_h_1 = math.ceil(padding_h)
                p_h_2 = math.floor(padding_h)

            if padding_w % 2 == 0:
                p_w_1 = int(padding_w)
                p_w_2 = int(padding_w)
            else:
                p_w_1 = math.ceil(padding_w)
                p_w_2 = math.floor(padding_w)

            output_height = int((height + p_h_1 + p_h_2 - self.convolution_shape[1]) / self.stride_shape[0]) + 1
            output_width = int((width + p_w_1 + p_w_2 - self.convolution_shape[2]) / self.stride_shape[1]) + 1
            output_tensor = np.zeros((b, self.num_kernels, output_height, output_width))

            input_padded = np.pad(input_tensor, pad_width=((0, 0), (0, 0), (p_h_2, p_h_1), (p_w_2, p_w_1)),
                                  mode='constant', constant_values=0)

            for batch in range(b):
                for kernel in range(self.num_kernels):
                    for channel in range(c):
                        output_tensor[batch, kernel] = output_tensor[batch, kernel] + correlate2d(
                            input_padded[batch, channel], self.weights[kernel, channel], mode='valid')[
                                                                                      ::self.stride_shape[0],
                                                                                      :: self.stride_shape[1]]

                    output_tensor[batch, kernel] = output_tensor[batch, kernel] + self.bias[kernel]

        # In the forward method, right before the return statement:
        # print("Input shape:", input_tensor.shape)
        # print("Stride:", self.stride_shape)
        # print("Kernel shape:", self.convolution_shape)
        # print("Output shape:", output_tensor.shape)

        self.input_tensor = input_tensor

        return output_tensor

    # def initialize(self,weights_initializer, bias_initializer):
    #     self.weights = weights_initializer
    #     self.bias = bias_initializer

    def initialize(self, weights_initializer, bias_initializer):
        # Compute fan_in and fan_out
        fan_in = np.prod(self.convolution_shape)  # All dimensions of the kernel
        fan_out = self.num_kernels * np.prod(self.convolution_shape[1:])  # Exclude input channels from fan_out

        # Initialize weights and biases using the initializers
        self.weights = weights_initializer.initialize(self.weights.shape, fan_in, fan_out)
        self.bias = bias_initializer.initialize(self.bias.shape, fan_in, fan_out)

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    @property
    def gradient_weights(self):
        return self.g_weights

    @property
    def gradient_bias(self):
        return self.g_bias

    def backward(self, error_tensor):
        batch_size = np.shape(error_tensor)[0]
        num_channels = self.convolution_shape[0]

        error_t = np.zeros((batch_size, self.num_kernels, *self.input_tensor.shape[2:]))
        gradient_input = np.zeros((batch_size, num_channels, *self.input_tensor.shape[2:]))
        gradient_weight = np.zeros((self.num_kernels, *self.convolution_shape))
        gradient_bias = np.zeros(self.num_kernels)
        tempory_gradient = np.zeros((self.num_kernels, *self.convolution_shape))

        weights = np.fliplr(np.swapaxes(self.weights, 0, 1))

        for b in range(batch_size):
            for c in range(num_channels):
                if len(self.stride_shape) == 1:
                    error_t[:, :, ::self.stride_shape[0]] = error_tensor[b]
                else:
                    error_t[:, :, ::self.stride_shape[0], ::self.stride_shape[1]] = error_tensor[b]

                t_output = convolve(error_t[b], weights[c], 'same')
                t_output = t_output[t_output.shape[0] // 2]
                gradient_input[b, c] = t_output

            if len(self.stride_shape) == 1:
                gradient_bias = np.sum(error_tensor, axis=(0, 2))
                padding_width = ((0, 0), (self.convolution_shape[1] // 2, (self.convolution_shape[1] - 1) // 2))
            else:
                gradient_bias = np.sum(error_tensor, axis=(0, 2, 3))
                padding_width = ((0, 0),(self.convolution_shape[1] // 2, (self.convolution_shape[1] - 1) // 2),(self.convolution_shape[2] // 2, (self.convolution_shape[2] - 1) // 2))

            input_pad = np.pad(self.input_tensor[b], padding_width, mode='constant', constant_values=0)

            for k in range(self.num_kernels):
                for c in range(num_channels):
                    tempory_gradient[k, c] = correlate(input_pad[c], error_t[b][k], 'valid')
            gradient_weight += tempory_gradient

        self.g_weights, self.g_bias = gradient_weight, gradient_bias
        if self.optimizer is not None:
            self.weights = copy.deepcopy(self.optimizer).calculate_update(self.weights, self.g_weights)
            self.bias = copy.deepcopy(self.optimizer).calculate_update(self.bias, self.g_bias)

        return gradient_input

