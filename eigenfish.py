from classify.classify import Classifier
from process.process import proc


class Eigenfish:
    def __init__(self, shape, process=None, classifier=None):
        self.process = (lambda img_mat, shape=shape: proc(img_mat, shape)
                        if process is None else process)
        self.classifier = Classifier() if classifier is None else classifier

    def train(self, img_mat, labels):
        self.classifier.train(self.process(img_mat), labels)

    def classify(self, img_mat):
        return self.classifier.classify(self.process(img_mat))

    def cross_validate(self, img_mat, labels):
        predicted = self.classifier.classify(self.process(img_mat))
        return (sum([0 if i != j else 1
                     for i, j in zip(labels, predicted)]) / len(labels))
