import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        self.loss = 0
        self.epsi = np.finfo(np.float64).eps
        self.error_tensor = None
        self.prediction_tensor = None
        self.batch_size = None

        pass

    def forward(self, prediction_tensor, label_tensor):
        self.prediction_tensor = prediction_tensor
        self.batch_size, num_classes = prediction_tensor.shape
        self.loss = np.sum(-label_tensor * np.log((prediction_tensor + self.epsi)))

        return self.loss

    def backward(self, label_tensor):
        self.error_tensor = -label_tensor / (self.prediction_tensor + self.epsi)
        return self.error_tensor
