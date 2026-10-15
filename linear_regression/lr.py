import numpy as np

class LinearRegression:
    def __init__(self):
        self.slope = 1
        self.bias = 0

    def _forward_pass(self, input):
        return self.slope * input + self.bias

    