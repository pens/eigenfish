from .math import rpca
from .math import fft2_series


class Processor:
    def process(self, img_mat, shape):
        """
        Process img_mat to prepare it for training/classification.

        :param img_mat: Matrix with each column a flattened image.
        :param shape: Original (width, height) of each image.
        """
        return fft2_series(rpca(img_mat)[1], shape)
