import numpy as np

class LinearRegression:
    def __init__(self):
        self.slope = 1
        self.bias = 0

    def _forward_pass(self, input):
        return self.slope * input + self.bias

    def _loss(self, y_pred, y_true):
        return (y_true - y_pred) ^ 2

    def _backprop(self, input):
        self.slope_grad = NULL
        self.bias_grad = NULL

        dldy = 2
        dydm = input
        dydc = self.slope * input

        self.slope_grad = dldy * dydm
        self.bias_grad = dldy * dydc

    def 