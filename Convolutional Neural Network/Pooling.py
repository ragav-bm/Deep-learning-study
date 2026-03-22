import numpy as np
from src_to_implement.Layers.Base import BaseLayer

class Pooling(BaseLayer):
    def __init__(self, stride_shape, pool_shape):
        super().__init__()
        self.stride_shape = stride_shape
        self.pool_shape = pool_shape

    def forward(self, input_tensor):
        self.inp_shape = np.shape(input_tensor)

        out_h = (np.shape(input_tensor)[2] - self.pool_shape[0]) // self.stride_shape[0] + 1
        out_w = (np.shape(input_tensor)[3] - self.pool_shape[1]) // self.stride_shape[1] + 1

        pooled_output = np.zeros((np.shape(input_tensor)[0], np.shape(input_tensor)[1], out_h, out_w))
        self.max_indices_x = np.zeros_like(pooled_output, dtype=int)
        self.max_indices_y = np.zeros_like(pooled_output, dtype=int)

        for row in range(out_h):
            for col in range(out_w):
                start_row = row * self.stride_shape[0]
                start_col = col * self.stride_shape[1]
                end_row = start_row + self.pool_shape[0]
                end_col = start_col + self.pool_shape[1]

                input_tensor1 = np.array(input_tensor)
                pooling_window = input_tensor1[:, :, start_row:end_row, start_col:end_col].reshape(
                    np.shape(input_tensor)[0], np.shape(input_tensor)[1], -1)

                max_values = np.max(pooling_window, axis=2)
                max_positions = np.argmax(pooling_window, axis=2)
                max_indices_row = max_positions // self.pool_shape[1]
                max_indices_col = max_positions % self.pool_shape[1]

                pooled_output[:, :, row, col] = max_values
                self.max_indices_x[:, :, row, col] = max_indices_row
                self.max_indices_y[:, :, row, col] = max_indices_col

        return pooled_output

    def backward(self, error_tensor):
        grad_tensor = np.zeros(self.inp_shape)
        for row in range(error_tensor.shape[2]):
            for col in range(error_tensor.shape[3]):
                start_row = row * self.stride_shape[0]
                start_col = col * self.stride_shape[1]
                # end_row = start_row + self.pool_shape[0]
                # end_col = start_col + self.pool_shape[1]

                for batch in range(error_tensor.shape[0]):
                    for channel in range(error_tensor.shape[1]):
                        max_row = start_row + self.max_indices_x[batch, channel, row, col]
                        max_col = start_col + self.max_indices_y[batch, channel, row, col]
                        grad_tensor[batch, channel, max_row, max_col] += error_tensor[batch, channel, row, col]

        return grad_tensor
