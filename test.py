import numpy
import os
import shutil
import unittest
from eigenfish import *
from process.process import *
from classify.classify import *
from util import *


class EmptyProcessor(Processor):
    def process(self, img_mat, shape):
        return img_mat

class EmptyClassifier(Classifier):
    def __init__(self):
        pass

    def train(self, data, labels):
        pass

    def classify(self, data):
        return ["none"]

    def cross_validate(self, data, labels):
        return 0.0

    def save(self, filename):
        pass

    def load(self, filename):
        pass

class TestEigenfish(unittest.TestCase):
    def setUp(self):
        self.shape = (10, 10)
        self.mat = numpy.hstack((numpy.ones((100, 10), 'F'),
                                 numpy.zeros((100, 10), 'F')))
        self.labels = (["ones" for i in range(10)] +
                       ["zeroes" for i in range(10)])
        self.test_mat = numpy.hstack((numpy.zeros((100, 3), 'F'),
                                      numpy.ones((100, 3), 'F')))
        self.test_labels = (["zeroes" for i in range(3)] +
                            ["ones" for i in range(3)])
        self.ef = Eigenfish(self.shape)
        os.mkdir("test/")

    def tearDown(self):
        shutil.rmtree("test/")

    def test_train_classify(self):
        self.ef.train(self.mat, self.labels)
        res = self.ef.classify(self.test_mat)
        for i in range(len(self.test_labels)):
            self.assertEqual(self.test_labels[i], res[i])

    def test_cross_validate(self):
        self.ef.train(self.mat, self.labels)
        pct_correct = self.ef.cross_validate(self.test_mat, self.test_labels)
        self.assertEqual(pct_correct, 1.0)

    def test_save_load(self):
        self.ef.train(self.mat, self.labels)
        res1 = self.ef.classify(self.test_mat)

        self.ef.save('test/temp.pkl')
        self.ef = Eigenfish(self.shape)
        self.ef.load('test/temp.pkl')

        res2 = self.ef.classify(self.test_mat)
        for i in range(len(res1)):
            self.assertEqual(res1[i], res2[i],
                             'Results not equal after save/load')
        ef = Eigenfish(self.shape, 'test/temp.pkl')
        res3 = ef.classify(self.test_mat)
        for i in range(len(res1)):
            self.assertEqual(res1[i], res3[i],
                             'Constructor training file load failed')

    def test_custom_modules(self):
        self.ef = Eigenfish(self.shape, None, EmptyProcessor, EmptyClassifier)
        self.ef.train(self.mat, self.labels)
        res = self.ef.classify(self.test_mat)
        self.assertEqual(["none"], res)

    def test_load_img_mat(self):
        filenames = ["test_data/%d.jpg" % i for i in range(4)]
        img_mat, shape = load_img_mat(filenames)
        self.assertEqual(img_mat.shape, (shape[0] * shape[1], 4))


class TestClassify(unittest.TestCase):
    def setUp(self):
        self.mat = numpy.hstack((numpy.ones((100, 10), 'F'),
                                 numpy.zeros((100, 10), 'F')))
        self.labels = (["ones" for i in range(10)] +
                       ["zeroes" for i in range(10)])
        self.test_mat = numpy.hstack((numpy.zeros((100, 3), 'F'),
                                      numpy.ones((100, 3), 'F')))
        self.test_labels = (["zeroes" for i in range(3)] +
                            ["ones" for i in range(3)])
        self.classifier = Classifier()
        os.mkdir("test/")

    def tearDown(self):
        shutil.rmtree("test/")

    def test_train_classify(self):
        self.classifier.train(self.mat, self.labels)
        res = self.classifier.classify(self.test_mat)
        for i in range(len(self.test_labels)):
            self.assertEqual(self.test_labels[i], res[i])

    def test_cross_validate(self):
        self.classifier.train(self.mat, self.labels)
        pct_correct = (
            self.classifier.cross_validate(self.test_mat, self.test_labels))
        self.assertEqual(pct_correct, 1.0)

    def test_save_load(self):
        self.classifier.train(self.mat, self.labels)
        res1 = self.classifier.classify(self.test_mat)

        self.classifier.save('test/temp.pkl')
        self.classifier = Classifier()
        self.classifier.load('test/temp.pkl')

        res2 = self.classifier.classify(self.test_mat)
        for i in range(len(res1)):
            self.assertEqual(res1[i], res2[i])


class TestProcess(unittest.TestCase):
    def setUp(self):
        self.shape = (10, 10)
        self.mat = numpy.random.random_sample((100, 10))

    def test_process(self):
        processor = Processor()
        proc_mat = processor.process(self.mat, self.shape)

    def test_rpca(self):
        l, s = rpca(self.mat)
        mat2 = l + s
        for i in range(self.mat.size):
            self.assertAlmostEqual(self.mat.flat[i], mat2.flat[i], 2)

    def test_fft2_series(self):
        fft = fft2_series(self.mat, self.shape)

if __name__ == "__main__":
    unittest.main()
