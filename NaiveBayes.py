import numpy as np
class NaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mean = {}
        self.var = {}
        self.priors = {}

        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = np.mean(X_c, axis=0)
            self.var[c] = np.var(X_c, axis=0) + 1e-6
            self.priors[c] = X_c.shape[0] / X.shape[0]

    def predict(self, X):
        predictions = []
        for x in X:
            posteriors = []
            for c in self.classes:
                prior = np.log(self.priors[c])
                likelihood = -0.5 * np.sum(np.log(2 * np.pi * self.var[c]))
                likelihood -= 0.5 * np.sum(((x - self.mean[c])**2) / self.var[c])
                posteriors.append(prior + likelihood)
            predictions.append(self.classes[np.argmax(posteriors)])
        return np.array(predictions)

# Example
X = np.array([[1,2],[2,1],[3,2],[6,5],[7,8],[8,6]])
y = np.array([0,0,0,1,1,1])
nb = NaiveBayes()
nb.fit(X,y)
print(nb.predict([[2,2],[7,7]]))
