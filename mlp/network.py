import numpy as np
from .activations import Softmax

class LinearLayer:
    def __init__(self, input_size, output_size, learning_rate=0.01):
        limit = np.sqrt(6 / (input_size + output_size))
        self.weights = np.random.uniform(-limit, limit, (input_size, output_size))
        self.bias = np.zeros((1, output_size))
        self.lr = learning_rate

    def forward(self, input_data):
        self.input = input_data
        return np.dot(self.input, self.weights) + self.bias

    def backward(self, grad_output):
        grad_weights = np.dot(self.input.T, grad_output)
        grad_bias = np.sum(grad_output, axis=0, keepdims=True)
        grad_input = np.dot(grad_output, self.weights.T)
        
        # A atualização dos pesos (SGD) está acoplada aqui na sua implementação
        self.weights -= self.lr * grad_weights
        self.bias -= self.lr * grad_bias
        
        return grad_input

class MLP:
    def __init__(self):
        self.layers = []
        self.activations = []

    def add_layer(self, layer, activation=None):
        self.layers.append(layer)
        self.activations.append(activation)

    def forward(self, x):
        for layer, activation in zip(self.layers, self.activations):
            x = layer.forward(x)
            if activation: x = activation.forward(x)
        return x

    def backward(self, grad):
        for layer, activation in reversed(list(zip(self.layers, self.activations))):
            if activation and not isinstance(activation, Softmax):
                grad = activation.backward(grad)
            grad = layer.backward(grad)
        return grad