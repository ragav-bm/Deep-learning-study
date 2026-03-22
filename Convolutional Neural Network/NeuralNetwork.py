import copy
from copy import deepcopy

class NeuralNetwork:
    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.optimizer = optimizer
        self.weights_initializer = deepcopy(weights_initializer)
        self.bias_initializer = deepcopy(bias_initializer)
        self.list = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.loss = []

    def forward(self):
        self.input_tensor, self.label_tensor = self.data_layer.next()
        for l in self.layers:
            self.input_tensor = l.forward(self.input_tensor)
        loss = self.loss_layer.forward(self.input_tensor, self.label_tensor)
        return loss

    def backward(self):
        error_tensor = self.loss_layer.backward(self.label_tensor)
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)

    def append_layer(self, layer):
        if layer.trainable == True:
            layer.initialize(self.weights_initializer,self.bias_initializer)
            layer.optimizer = copy.deepcopy(self.optimizer)
        self.layers.append(layer)

    def train(self, iterations):
        for n in range(iterations):
            loss = self.forward()
            self.loss.append(loss)
            self.backward()

    def test(self, input_tensor):
        for l in self.layers:
            input_tensor = l.forward(input_tensor)
            # print(l)
        return input_tensor