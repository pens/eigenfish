import math
import numpy
import scipy.sparse.linalg


def rpca(image_mat):
    """
    Performs Robust Principle Component Analysis on image_mat.

    :returns: Low-rank, sparse parts of image_mat
    """
    m = image_mat.shape[0]
    lam = 1 / math.sqrt(m)
    tol = 1e-7
    max_iter = 40
    norm_two = numpy.linalg.norm(image_mat, 2)
    norm_inf = numpy.linalg.norm(image_mat.flatten(), numpy.inf) / lam
    norm_fro = numpy.linalg.norm(image_mat, 'fro')
    dual_norm = max(norm_two, norm_inf)
    y = image_mat / dual_norm

    a_hat = numpy.empty(image_mat.shape)
    e_hat = numpy.empty(image_mat.shape)
    mu = 1.25 / norm_two
    mu_bar = mu * 1e7
    rho = 1.5

    for i in range(max_iter):
        temp = image_mat - a_hat + 1 / mu * y
        e_hat = (numpy.maximum(temp - lam / mu, 0) +
                 numpy.minimum(temp + lam / mu, 0))

        u, sigma, vt = (scipy.sparse.linalg.svds(
            numpy.asarray(image_mat - e_hat + 1 / mu * y)))
        svp = (sigma > 1 / mu).sum()

        a_hat = (u[:, -svp:].dot(numpy.diag(sigma[-svp:] - 1 / mu)).dot(
            vt[-svp:, :]))

        z = image_mat - a_hat - e_hat
        y += mu * z
        mu = min(mu * rho, mu_bar)

        if (numpy.linalg.norm(z, 'fro') / norm_fro) < tol:
            break

    return a_hat, e_hat


def fft2_series(img_mat, shape):
    """
    For each column in img_mat, img_mat[:, i] the fft2 modes are extracted and
    placed into the corresponding column of the returned matrix.

    :param img_mat: Matrix to process.
    :param shape: Original (width, height) of each column of img_mat.
    :returns: New numpy.ndarray matrix, where return[:, i] is the fft2 modes of
        img_mat[:, i].
    """
    fourier = numpy.empty(img_mat.shape)
    for i in range(img_mat.shape[1]):
        fourier[:, i] = (
            numpy.abs(numpy.fft.fft2(img_mat[:, i].reshape(shape))).flatten())
    return fourier
