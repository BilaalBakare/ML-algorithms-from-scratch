import numpy as np

class nn:
    def __init__(self, layers: list):
        self.layers = layers

        self.weights = []
        self.biases = []

        for i in range(len(self.layers)-1):
            shape = self.layers[i], self.layers[i+1]

            self.weights.append(np.random.randn(*shape))
            self.biases.append(np.zeros(shape[1]))

    def _forward_pass(self, input: np.ndarray) -> np.ndarray: 
        self.layer_outputs = []
        self.layer_outputs_activated = []

        self.layer_outputs_activated.append(input)

        for i in range(len(self.weights)):
            current = input @ self.weights[i] + self.biases[i]
            self.layer_outputs.append(current)

            activated = self._softmax(current) if i == (len(self.weights) - 1) else self._relu(current)
            self.layer_outputs_activated.append(activated)
            
            input = activated

        return self.layer_outputs_activated[-1]

    def _softmax(self, layer_output: np.ndarray):
        e_x = np.exp(layer_output - np.max(layer_output, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def _relu(self, layer_output: np.ndarray):
        return np.maximum(0, layer_output)
    
    def _relu_derivative(self, layer_output: np.ndarray) -> np.ndarray:
        return (layer_output > 0).astype(float)

    def _loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        # categorical cross entropy
        y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)

        loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        return loss
    
    def _backpropagation(self, y_true: np.ndarray):
        error_signal = self.layer_outputs_activated[-1] - y_true

        self.weight_gradients = []
        self.bias_gradients = []
        
        for i in range(len(self.weights)-1, -1, -1):
            weigth_gradient = self.layer_outputs_activated[i].T @ error_signal 
            self.weight_gradients.append(weigth_gradient)
            bias_gradient = np.sum(error_signal, axis=0)
            self.bias_gradients.append(bias_gradient)

            if i == 0:
                break 

            error_signal_active = error_signal @ self.weights[i].T
            error_signal_normal = error_signal_active * self._relu_derivative(self.layer_outputs[i-1])
            error_signal = error_signal_normal

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int, learning_rate: int, batch_size=32):
        y_train = np.eye(10)[y_train]
        for epoch in range(epochs):
            indices = np.random.permutation(len(X_train))

            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            train_set_size = len(X_shuffled)

            for batch in range(0, train_set_size, batch_size):
                X_batch = X_shuffled[batch:batch+batch_size]
                y_batch = y_shuffled[batch:batch+batch_size]

                y_pred = self._forward_pass(X_batch)
                loss = self._loss(y_pred, y_batch) 
                self._backpropagation(y_batch)

                for i in range(len(self.weights)):
                    self.weights[i] -= learning_rate * self.weight_gradients[-(i+1)]
                    self.biases[i]  -= learning_rate * self.bias_gradients[-(i+1)]

            print(f'Epoch {epoch} loss = {loss}')

    def predict(self, )
