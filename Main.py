import numpy as np
import pandas as pd
from OneVsAllClassifier import OneVsAllClassifier

data_file_name = 'owls.csv'
test_train_split = 0.33

data = pd.read_csv(data_file_name)

classifier = OneVsAllClassifier(data, test_train_split=0.66)
classifier.trainClassifiers()
classifier.run_tests()

# X_test = test_data[:, :-1]
# y_test = test_data[:, -1]

