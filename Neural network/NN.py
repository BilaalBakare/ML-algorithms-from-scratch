import numpy as np

class nn:
    def __init__(self, layers: list):
        self.layers = layers

        self.weights = []
        self.biases = []

        for i in range(len(layers) -1):
            shape = layers[i], layers[i+1]

            self.weight.append(np.random.randn(*shape))
            self.biases.append(np.zeros(shape[1]))