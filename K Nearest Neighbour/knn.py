import numpy as np
import pandas as pd
from collections import Counter

class Knn:
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X_train, y_train):
        self.x_train = np.array(X_train)
        self.y_train = np.array(y_train)

    def predict(self, X_test):
        y_pred = []

        for i in X_test:
            pred = predict_single(i)
            y_pred.append(pred)

        return y_pred
    
    def predict_single(self, X):
        pair_list = []

        for i in self.X_train:
            sample_pair = distance(X, self.X_train[i]), self.y_train[i]
            pair_list.append(sample_pair)
        
        pair_list.sort(lambda x: x[0])

        # fke = first K elements
        fke = pair_list[:self.k]

        count = Counter(category[1] for category in fke)

        prediction = count.most_common(1)[0]

        return prediction
    
    def distance(self, X1, xX):
        X1 = np.array(X1)
        X2 = np.array(X2)

        dist = np.sqrt(np.sum((X1-X2)**2))
        return dist
    
    def __repr__(self):
        return f"KNN(k={self.k})"