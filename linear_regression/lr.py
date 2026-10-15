class LinearRegression:
    def __init__(self):
        self.slope = 1
        self.bias = 0

    def _forward_pass(self, input):
        return self.slope * input + self.bias

    def _loss(self, y_pred, y_true):
        return (y_true - y_pred) ^ 2

    def _backprop(self, input):
        self.slope_grad = None
        self.bias_grad = None

        dldy = 2
        dydm = input
        dydc = self.slope * input

        self.slope_grad = dldy * dydm
        self.bias_grad = dldy * dydc

    def fit(self, x, y_true, lr):

        counter = 1

        for input, output in zip(x, y_true):
            y_pred = self._forward_pass(input)
            loss = self._loss(y_pred, output)
            self._backprop(input)

            print(f'Epoch {counter}: Loss = {loss}')

            self.slope -= lr * self.slope_grad
            self.bias -= lr * self.bias_grad