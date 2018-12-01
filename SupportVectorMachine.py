import numpy as np


class SupportVectorMachine:
    def __init__(self):
        pass

    def fit(self):
        # Convert y classes to +1, -1
        binary_y = np.zeros((len(self.y), 1))
        neg_class = self.classes[0]
        pos_class = self.classes[1]
        for i in range(0, len(self.y)):
            if self.y[i] == neg_class:
                binary_y[i] = -1
            elif self.y[i] == pos_class:
                binary_y[i] = 1

        # Add bias to X for stochastic gradient descent
        X_bias = np.empty((len(self.X), 1))
        X_bias.fill(-1)
        biased_X = np.append(self.X, X_bias, axis=1)

        w = np.zeros(len(biased_X[0]))
        eta = 1
        epochs = 10000

        for epoch in range(1, epochs):
            for i in range(0, len(biased_X)):
                if (binary_y[i] * np.dot(biased_X[i], w)) < 1:
                    w = w + eta * ((biased_X[i] * binary_y[i]) + (-2 * (1 / epoch) * w))
                else:
                    w = w + eta * (-2 * (1 / epoch) * w)

        self.w = w[:-1]
        self.b = w[-1]

    def predict(self, features):
        result = np.sign(np.dot(np.array(features), self.w) + self.b)
        if result == -1:
            return self.classes[0]
        elif result == 1:
            return self.classes[1]
        return "Error"

    def getclasses(self, y):
        return np.unique(y)

    def train(self, X, y):
        self.X = X
        self.y = y
        self.classes = self.getclasses(y)

        self.fit()
