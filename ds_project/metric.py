import numpy as np


class Metrics:
    def __init__(self, actual, prediction):
        self.actual = np.array(actual)
        self.prediction = np.array(prediction)

    def wape(self):
        actual = self.actual
        prediction = self.prediction

        metric = np.sum(np.abs(actual - prediction)) / np.sum(actual)

        return metric

    def mae(self):
        actual = self.actual
        prediction = self.prediction

        metric = np.mean(np.abs(actual - prediction))

        return metric