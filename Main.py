import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import pandas as pd

style.use('ggplot')

data_file_name = 'owls_test.csv'
index_of_class = 4


class SupportVectorMachine:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    # train
    def fit(self):
        # Add bias to X for stochastic gradient descent
        X_bias = np.empty((len(self.X), 1))
        X_bias.fill(-1)
        biased_X = np.append(self.X, X_bias, axis=1)

        w = np.zeros(len(biased_X[0]))
        eta = 1
        epochs = 10000

        for epoch in range(1, epochs):
            for i in range(0, len(biased_X)):
                if (self.y[i] * np.dot(biased_X[i], w)) < 1:
                    w = w + eta * ((biased_X[i] * self.y[i]) + (-2 * (1 / epoch) * w))
                else:
                    w = w + eta * (-2 * (1 / epoch) * w)

        self.w = w[:-1]
        self.b = w[-1]

    def predict(self, features):
        return np.sign(np.dot(np.array(features), self.w) + self.b)


data = pd.read_csv(data_file_name)

data = data.values
np.random.shuffle(data)

train_data_len = int((2 * len(data)) / 3)

train_data = data[:train_data_len]
test_data = data[train_data_len:]

# Now I want to splilt the data up into two classes, -1 and 1
X_train = train_data[:, :-1]
y_train = train_data[:, -1]

# Change out y's to +1, -1
for i in range(0, len(y_train)):
    if y_train[i] == "LongEaredOwl":
        y_train[i] = -1
    else:
        y_train[i] = 1

svm = SupportVectorMachine(X_train, y_train)
svm.fit()

X_test = test_data[:, :-1]
y_test = test_data[:, -1]

for i in range(0, len(X_test)):
    print("Test ", i)
    print("Predicted: ", svm.predict(X_test[i]))
    print("Actual: ", y_test[i], "\n")

#svm.visualize()
