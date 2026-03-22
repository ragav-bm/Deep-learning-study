import numpy as np
from Layers import Base
from Layers.TanH import TanH
from Layers.FullyConnected import FullyConnected

class RNN(Base.BaseLayer):

    def __init__(self, input_size, hidden_size, output_size):

        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.trainable = True
        self._memorize = False
        
        self.tanh = TanH()

        self.w_xhh = FullyConnected(input_size+hidden_size,hidden_size)
        self.w_hy = FullyConnected(hidden_size, output_size)

        
        
        self.g_w_xhh = np.zeros((self.input_size+self.hidden_size + 1, self.hidden_size))
        self.g_w_xhy = np.zeros((self.hidden_size + 1, self.output_size))

        self.hidden_states = None
        self.last_hidden_states = None
        self.optimizer = None

        self.weights = self.w_xhh.weights
        
    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, value):
        self._memorize = value

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        self._weights = weights

    @property
    def gradient_weights(self):
        return self.g_w_xhh

    def initialize(self, weights_initializer, bias_initializer):
        self.w_xhh.initialize(weights_initializer, bias_initializer)
        
        self.w_hy.initialize(weights_initializer, bias_initializer)
        

    def forward(self, input_tensor):
        b, _ = input_tensor.shape

        if self._memorize:
            if self.hidden_states is None:
                self.hidden_states = np.zeros((b+ 1, self.hidden_size))
            else:
                self.hidden_states[0] = self.last_hidden_states
        else:
            self.hidden_states = np.zeros((b + 1, self.hidden_size))

        next_input_tensor = np.zeros((b, self.output_size))

        for t in range(b):
            x_t = np.concatenate((self.hidden_states[t][np.newaxis, :], input_tensor[t][np.newaxis, :]), axis=1)

            self.hidden_states[t + 1] = self.tanh.forward(self.w_xhh.forward(x_t))

            next_input_tensor[t] = self.w_hy.forward(self.hidden_states[t + 1][np.newaxis, :])

        self.last_hidden_states = self.hidden_states[-1]
        
        
        self.error_b =b
        self.input_tensor = input_tensor
        

        return next_input_tensor

    def backward(self, error_tensor):
        
        self.g_wh = np.zeros((self.hidden_size + self.input_size + 1, self.hidden_size))
        self.next_error_tensor = np.zeros((self.error_b, self.input_size))

        gradient_hidden_activation = 1 - self.hidden_states[1::] ** 2
        hidden_states_error = np.zeros((1, self.hidden_size))

        
        for t in reversed(range(self.error_b)):
            next_error_tensor = self.w_hy.backward(error_tensor[t][np.newaxis, :])

            gradient_yh = (hidden_states_error + next_error_tensor)
            hidden_states_error = self.w_xhh.backward( gradient_hidden_activation[t] * gradient_yh)[:, 0:self.hidden_size]
            self.next_error_tensor[t] = self.w_xhh.backward( gradient_hidden_activation[t] * gradient_yh)[:, self.hidden_size:(self.hidden_size + self.input_size + 1)]

            self.g_wh = self.g_wh+  self.w_xhh.gradient_weights
            self.g_w_xhy = self.g_w_xhy + self.w_hy.gradient_weights

        if self.optimizer is not None:
            self.w_hy.weights = self.optimizer.calculate_update(self.w_hy.weights, self.g_w_xhy)
            self.w_xhh.weights = self.optimizer.calculate_update(self.w_xhh.weights, self.g_wh)

        return self.next_error_tensor



