from sklearn import preprocessing
from sklearn import svm
from sklearn.externals import joblib


class Classifier:
    def __init__(self):
        self.svc = svm.SVC()

    def train(self, data, labels):
        self.svc.fit(preprocessing.scale(data.T), labels)

    def classify(self, data):
        return self.svc.predict(preprocessing.scale(data.T))

    def load(self, filename):
        self.svc = joblib.load(filename)

    def save(self, filename):
        joblib.dump(self.svc, filename)