import numpy as np
import pandas as pd
from collections import Counter

class Knn:
    def __init__(self, k=5):
        self.k = k
    
    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

    def predict(self, X_test):
        y_pred = []

        for i in X_test:
            pred = self._predict_single(i)
            y_pred.append(pred)

        return np.array(y_pred)
    
    def _predict_single(self, X):
        distance = np.sqrt(np.sum((self.X_train - X) ** 2, axis=1))
        distance_sorted_sliced = np.argsort(distance)[:self.k]
        labels = self.y_train[distance_sorted_sliced]

        count = Counter(labels)
        prediction = count.most_common(1)[0][0]

        return prediction
    
    def __repr__(self):
        return f"KNN(k={self.k})"