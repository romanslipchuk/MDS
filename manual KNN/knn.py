import numpy as np
from collections import Counter

class KNN():

    def __init__(self, k=5, metric='euclidian'):
        self.k = k
        self.metric = metric


    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        return

    def predict(self,x_test):
        predictions = []
        for j in range(len(x_test)):

            distances = []
            targets = []

            for i in range(len(self.X_train)):
                if self.metric == 'euclidian':
                    distances.append([np.sqrt(np.sum(np.square(x_test[j] - self.X_train[i]))), i])
                if self.metric == 'manhattan':
                    distances.append([np.sum(np.abs(x_test[j] - self.X_train[i])), i])

            distances = sorted(distances)

            for i in range(self.k):
                index = distances[i][1]
                targets.append((self.y_train[index]))

            predictions.append(Counter(targets).most_common(1)[0][0])
        return predictions
