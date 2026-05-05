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

    def _forward_pass(self, input: list): 
        self.layer_outputs = []
        self.layer_outputs_activated = []

        for i in range(len(self.weights)):
            current = input @ self.weights[i] + self.biases[i]
            self.layer_outputs.append(current)

            activated = self._sigmoid(current) if i == (len(self.weights) - 1) else self._relu(current)
            self.layer_outputs_activated.append(activated)
            
            input = activated

        return self.layer_outputs_activated[-1]

    def _sigmoid(self, layer_output: np.ndarray) -> np.ndarray: 
        return 1 / (1 + np.exp(-layer_output))

    def _relu(self, layer_output: np.ndarray) -> np.ndarray:
        return np.maximum(0, layer_output)

    def _loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)
        numerator = -((y_true * np.log(y_pred)) + ((1-y_true) * np.log(1-y_pred)))
        denominator = len(y_true)

        loss = np.sum(numerator) / denominator
        return loss
