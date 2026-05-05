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

    def forward_pass(self, input: list): 
        self.layer_outputs = []
        self.layer_outputs_activated = []

        for i in range(len(self.weights)):
            current = input @ self.weights[i] + self.biases[i]
            self.layer_outputs.append(current)

            activated = self._sigmoid(current) if i == (len(self.weights) - 1) else self._relu(current)
            self.layer_outputs_activated.append(activated)
            
            input = activated

        return self.layer_outputs_activated[-1]

    def _sigmoid(self):
        pass

    def _relu(self):
        pass
