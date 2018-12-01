import numpy as np
import pandas as pd
from SupportVectorMachine import SupportVectorMachine

data_file_name = 'owls_test.csv'
test_train_split = 0.33

data = pd.read_csv(data_file_name)

# Shuffle the data and split into test and train
data = data.values
np.random.shuffle(data)
train_data_len = int(len(data) * (1 - test_train_split))
train_data = data[:train_data_len]
test_data = data[train_data_len:]

# Now I want to split the data up into X and y
X_train = train_data[:, :-1]
y_train = train_data[:, -1]

svm = SupportVectorMachine()
svm.train(X_train, y_train)

X_test = test_data[:, :-1]
y_test = test_data[:, -1]

for i in range(0, len(X_test)):
    print("Test ", i)
    print("Predicted: ", svm.predict(X_test[i]))
    print("Actual: ", y_test[i], "\n")
