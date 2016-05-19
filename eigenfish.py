from classify.classify import Classifier
from process.process import Processor


class Eigenfish:
    def __init__(self, training_file=None, processor=None, classifier=None):
        if training_file is not None:
            self.load(training_file)
        self.processor = Processor if processor is None else processor
        self.classifier = Classifier if classifier is None else classifier

    def train(self, img_mat, shape, label_arr):
        self.classifier.train(self.processor.process(img_mat, shape), label_arr)

    def classify(self, img_mat, shape):
        return self.classifier.classify(self.processor.process(img_mat, shape))

    def cross_validate(self, img_mat, shape, label_arr):
        pred = self.classifier.classify(self.processor.process(img_mat, shape))
        return (sum([0 if i != j else 1
                     for i, j in zip(label_arr, pred)]) / len(label_arr))

    def load(self, filename):
        self.classifier.load(filename)

    def save(self, filename):
        self.classifier.save(filename)
