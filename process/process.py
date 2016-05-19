from .math import rpca
from .math import fft2_series


class Processor:
    def process(self, img_mat, shape):
        return fft2_series(rpca(img_mat)[1], shape)
