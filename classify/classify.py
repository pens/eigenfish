import scipy.sparse.linalg
from sklearn import preprocessing
from sklearn import svm
from sklearn.externals import joblib


class Classifier:
    def __init__(self):
        """
        Initialize untrained classifier.
        """
        self.svc = svm.SVC()
        #self.svc = svm.SVC(kernel='linear')
        self.use_pca = False
        self.rank = 1

    def train(self, data, labels):
        """
        Trains current classifier with matrix data and labels, where labels[i]
        describes data[:, i].

        :param data: Matrix of data, where each column is a separate sample.
        :param labels: List of labels, each corresponding to a column of data.
        """
        if self.use_pca:
            u, s, _ = scipy.sparse.linalg.svds(data)
            self.svc.fit(
                preprocessing.scale(u[:, :self.rank+1].T.dot(data).T), labels)
        else:
            self.svc.fit(preprocessing.scale(data.T), labels)

    def classify(self, data):
        """
        Classifies data based on current model.

        :param data: Matrix with each column a different sample.
        :returns: List of predictions, where return[i] describes data[:, i].
        """
        if self.use_pca:
            u, s, _ = scipy.sparse.linalg.svds(data)
            self.svc.predict(
                preprocessing.scale(u[:, :self.rank+1].T.dot(data).T))
        else:
            return self.svc.predict(preprocessing.scale(data.T))

    def cross_validate(self, data, labels):
        """
        Cross-validates trained model against data with labels.

        :param data: Matrix with each column a different sample.
        :param labels: List of labels, each corresponding to a column of data.
        :returns: Percent labels the same.
        """
        pred = self.classify(data)
        return (sum([0 if i != j else 1
                for i, j in zip(labels, pred)]) / len(labels))

    def load(self, filename):
        """
        Loads trained model from file, overwriting current model. Do not use on
        training files you did not create.

        :param filename: Name of file to load.
        """
        self.svc = joblib.load(filename)

    def save(self, filename):
        """
        Saves trained model to filename.

        :param filename: Name of file to save model as.
        """
        joblib.dump(self.svc, filename)
