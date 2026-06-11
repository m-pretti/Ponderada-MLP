import numpy as np

class ReLU:
    def forward(self, x):
        self.input = x
        return np.maximum(0, x)

    def backward(self, grad_output):
        return grad_output * (self.input > 0)

class Softmax:
    def forward(self, x):
        # Subtraímos o max para estabilidade numérica (evitar overflow)
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.output = exps / np.sum(exps, axis=1, keepdims=True)
        return self.output

    def backward(self, target):
        pass