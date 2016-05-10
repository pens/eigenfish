import numpy
from .rpca import rpca


def rem_background(img_mat):
    return rpca(img_mat)[1]


def extract_modes(img_mat, shape):
    data = numpy.empty(img_mat.shape)
    for i in range(img_mat.shape[1]):
        data[:, i] = (
            numpy.abs(numpy.fft.fft2(img_mat[:, i].reshape(shape))).flatten())
    return data


def proc(img_mat, shape):
    return extract_modes(rem_background(img_mat), shape)
