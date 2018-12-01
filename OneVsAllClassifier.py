import numpy as np
import pandas as pd
from SupportVectorMachine import SupportVectorMachine


class OneVsAllClassifier:

    def __init__(self, data, test_train_split=0.9):
        self.data, self.train_data, self.test_data = self.spiltData(data, test_train_split)
        self.X_train, self.y_train = self.xySplit(self.train_data)
        self.X_test, self.y_test = self.xySplit(self.test_data)
        self.classes = self.getclasses(self.y_train)

        self.classification_machines = self.setupClassifiers(self.classes)

    def spiltData(self, data, test_train_split):
        data = data.values
        np.random.shuffle(data)
        train_data_len = int(len(data) * (1 - test_train_split))
        train_data = data[:train_data_len]
        test_data = data[train_data_len:]

        return data, train_data, test_data

    def getclasses(self, y):
        return np.unique(y)

    def xySplit(self, data):
        X = data[:, :-1]
        y = data[:, -1]

        return X, y

    def setupClassifiers(self, classes):
        classification_machines = [[]]

        for oneclass in classes:
            svm = SupportVectorMachine()
            classification_machines.append([oneclass, svm])

        classification_machines.pop(0)
        return classification_machines

    def trainClassifiers(self):
        print("Training SVM's")
        for oneclass, svm in self.classification_machines:
            X_local = np.copy(self.X_train)
            y_local = np.copy(self.y_train)

            y_local[y_local != oneclass] = "Not" + oneclass

            svm.train(X_local, y_local)

            print(oneclass, " SVM Trained")

    def predict(self, features):
        predictions = []
        result = ""
        for oneclass, svm in self.classification_machines:
            predictions.append(svm.predict(features))

        for prediction in predictions:
            for oneclass in self.classes:
                if prediction == oneclass:
                    result += prediction

        return result

    def run_tests(self):
        correct_predictions = 0
        number_of_predictions = len(self.X_test)
        for i in range(0, number_of_predictions):
            prediction = self.predict(self.X_test[i])
            actual = self.y_test[i]
            if prediction == actual:
                correct_predictions += correct_predictions + 1

            print("Test ", i)
            print("Predicted: ", prediction)
            print("Actual: ", actual, "\n")

        print("\n**** Test Results ****" \
              "\nCorrect Predictions: ", correct_predictions, \
              "\nNumber of Predictions", number_of_predictions, \
              "\nAccuracy: ", correct_predictions/number_of_predictions)
