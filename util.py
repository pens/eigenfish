import numpy
import scipy.misc


def load_img_mat(files):
    """
    Loads all files as images in black and white, flattens them and returns them
    as a numpy.ndarray.

    :param files: List of image files to load. All should be of the same
        resolution.
    :return: Numpy.ndarray matrix with each column a flattened images.
    """
    shape = scipy.misc.imread(files[0], True).shape
    img_mat = numpy.empty((shape[0] * shape[1], len(files)))
    for i, file in enumerate(files):
        img_mat[:, i] = scipy.misc.imread(file, True).flatten()
    return img_mat, shape
