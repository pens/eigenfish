from sklearn import preprocessing
from sklearn import svm


class Classifier:
    def __init__(self):
        self.svc = svm.SVC()

    def train(self, data, labels):
        self.svc.fit(preprocessing.scale(data.T), labels)

    def classify(self, data):
        return self.svc.predict(preprocessing.scale(data.T))
