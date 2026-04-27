import numpy as np

class naive:
    def __init__(self):
        pass
    def fit(self, X_train, y_train):
        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # priors

        self._prior_dictionary = {}
        target_len = len(y_train)

        self.classes = np.unique(y_train)
        for aclass in self.classes:
            count = np.sum([y_train == aclass])
            prob = count / target_len
            self._prior_dictionary[aclass] = prob

        
        
        # likelihoods

        self._mean_dictionary = {}
        self._variance_dictionary = {}

        for aclass in self.classes:
            rows = X_train[y_train == aclass]

            mean = np.mean(rows, axis=0)
            self._mean_dictionary[aclass] = mean

            variance = np.var(rows, axis=0) + 1e-9
            self._variance_dictionary[aclass] = variance
        
            
    
    def predict(self, X_test):
        X_test = np.array(X_test)

        y_pred = []
        for i in X_test:
            class_probs = []
            for aclass in self.classes:
                likelihoods = np.sum(np.log(self._gaussian(i, self._mean_dictionary[aclass], self._variance_dictionary[aclass])))
                prior = self._prior_dictionary[aclass] 
                prob = np.log(prior) + likelihoods

                class_probs.append((aclass, prob))
            prediction = max(class_probs, key=lambda x: x[1])[0]
            y_pred.append(prediction)
        
        return np.array(y_pred)
            

    def _gaussian(self, x, mean, var):
        numerator   = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator