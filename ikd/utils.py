#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from scipy.spatial.distance import cdist
from sklearn.gaussian_process.kernels import Matern
from scipy import special, signal


def kernel_cov_generator(z: np.array, z_other=None, kernel="squared exponential", variance=1, length_scale=1, extra_kernel_hyperparam=None, show=False) -> np.array:
    """Kernel covariance matrix generator.

    Parameters
    ----------
    z : ndarray of shape (n_points, d_latent)
        Latents for building kernel covariance matrix.
    z_other : ndarray of shape (n_points_other, d_latent), optional
        Another set of latents, by default None. If not provided, compute kernel covariance matrix k(z, z); otherwise, compute kernel covariance matrix k(z, z_other).
    kernel : str, optional
        ["squared exponential" | "rational quadratic" | "gamma-exponential" | "matern"], by default "squared exponential".
    variance : int, optional
        Marginal variance, by default 1.
    length_scale : int, optional
        Length scale of the kernel, by default 1.
    extra_kernel_hyperparam : float, optional
        α in rational quadratic kernel; γ in gamma-exponential kernel; ν in Matérn kernel, by default None.
    show : bool, optional
        Visualization flag, by default False.

    Returns
    -------
    cov : ndarray of shape (n_points, n_points)
        Covariance matrix according to the latents.
    """

    if z_other is None:
        z_other = z.copy()

    if kernel == 'squared exponential':
        cov = variance * np.exp(-cdist(z, z_other, 'sqeuclidean') / (2 * length_scale**2))
    elif kernel == 'rational quadratic':
        cov = variance * (1 + cdist(z, z_other, 'sqeuclidean') / (2 * extra_kernel_hyperparam * length_scale**2))**(-extra_kernel_hyperparam)
    elif kernel == 'gamma-exponential':
        cov = variance * np.exp(-cdist(z, z_other)**extra_kernel_hyperparam / length_scale**extra_kernel_hyperparam)
    elif kernel == 'matern':
        # if nu == 1.5:
        #     pairwise_dist = np.sqrt(pairwise_dist2)
        #     cov = variance * (1 + np.sqrt(3) * pairwise_dist / length_scale) * np.exp(-np.sqrt(3) * pairwise_dist / length_scale)
        # elif nu == 2.5:
        #     pairwise_dist = np.sqrt(pairwise_dist2)
        #     cov = variance * (1 + np.sqrt(5) * pairwise_dist / length_scale + 5 * pairwise_dist2 / (3 * length_scale**2)) * np.exp(-np.sqrt(5) * pairwise_dist / length_scale)
        # else:
        #     print("Wrong")
        matern = Matern(length_scale=length_scale, nu=extra_kernel_hyperparam)
        cov = variance * matern(z, z_other)
    elif kernel == 'autoregressive':
        cov = variance * np.exp(-cdist(z, z_other, 'minkowski', p=1.) / length_scale)
    else:
        raise ValueError('No such kernel')
    if show is True:
        plt.figure()
        plt.matshow(cov)
        plt.colorbar()
        plt.title('True covariance matrix')
    return cov

def cov2dist2(cov: np.array, kernel="squared exponential", variance=1, length_scale=1, extra_kernel_hyperparam=None) -> np.array:
    """Convert the covariance matrix squared pairwise distance matrix.

    Parameters
    ----------
    cov : ndarray of shape (n_points, n_points)
        Covariance matrix.
    kernel : str, optional
        ["squared exponential" | "rational quadratic" | "gamma-exponential" | "matern"], by default "squared exponential".
    variance : int, optional
        Marginal variance, by default 1
    length_scale : int, optional
        Length scale of the kernel, by default 1.
    extra_kernel_hyperparam : float, optional
        α in rational quadratic kernel; γ in gamma-exponential kernel; ν in Matérn kernel, by default None.

    Returns
    -------
    pairwise_dist2 : ndarray of shape (n_points, n_points)
        Square of pairwise distance matrix induced from the covariance matrix.
    """

    cov_scaled = cov / variance
    if kernel == 'squared exponential':
        pariwise_dist2 = -np.log(cov_scaled) * 2 * length_scale**2
    elif kernel == 'rational quadratic':
        pariwise_dist2 = (cov_scaled**(-1/extra_kernel_hyperparam) - 1) * 2 * extra_kernel_hyperparam * length_scale**2
    elif kernel == 'gamma-exponential':
        pariwise_dist2 = ((-np.log(cov_scaled))**(1/extra_kernel_hyperparam) * length_scale)**2
    elif kernel == 'matern':
        if extra_kernel_hyperparam == 0.5:
            pariwise_dist2 = (-np.log(cov_scaled))**2
        elif extra_kernel_hyperparam == 1.5:
            n = cov.shape[0]
            pariwise_dist2 = np.zeros((n, n))
            for i in range(n):
                for j in range(i + 1, n):
                    root_result = root_scalar(lambda x: (1 + x) * np.exp(-x) - cov_scaled[i, j], x0=1, fprime=lambda x: - x * np.exp(-x))
                    if root_result.converged is False:
                        raise ValueError(f"Unable to identify the distance between {i} and {j}")
                    pariwise_dist2[i, j] = root_result.root
            pariwise_dist2 = (pariwise_dist2 * length_scale / np.sqrt(3))**2
            pariwise_dist2 += np.triu(pariwise_dist2, k=1).T
        elif extra_kernel_hyperparam == 2.5:
            n = cov.shape[0]
            pariwise_dist2 = np.zeros((n, n))
            for i in range(n):
                for j in range(i + 1, n):
                    root_result = root_scalar(lambda x: (1 + x + x**2/3) * np.exp(-x) - cov_scaled[i, j], x0=1, fprime=lambda x: - x / 3 * (1 + x) * np.exp(-x))
                    if root_result.converged is False:
                        raise ValueError(f"Unable to identify the distance between {i} and {j}")
                    pariwise_dist2[i, j] = root_result.root
            pariwise_dist2 = (pariwise_dist2 * length_scale / np.sqrt(5))**2
            pariwise_dist2 += np.triu(pariwise_dist2, k=1).T
        else:
            n = cov.shape[0]
            pariwise_dist2 = np.zeros((n, n))
            # matern = Matern(length_scale=1, nu=nu)
            def matern(x):
                if x == 0:
                    x += np.finfo(float).eps
                tmp = np.sqrt(2 * extra_kernel_hyperparam) * x
                return 2**(1 - extra_kernel_hyperparam) / special.gamma(extra_kernel_hyperparam) * tmp**extra_kernel_hyperparam * special.kv(extra_kernel_hyperparam, tmp)

            root_result = root_scalar(lambda x: matern(x) - 1e-5 / variance, x0=2, x1=3)
            if root_result.converged is False:
                raise ValueError(f"Unable to identify the upperbound of this matern kernel")
            else:
                upperbound = root_result.root + 0.5
            for i in range(n):
                for j in range(i+1, n):
                    root_result = root_scalar(lambda x: matern(x) - cov_scaled[i, j], x0=1, bracket=(0, upperbound))
                    if root_result.converged is False:
                        raise ValueError(f"Unable to identify the distance between {i} and {j}")
                    pariwise_dist2[i, j] = root_result.root
            pariwise_dist2 = pariwise_dist2**2                    
            pariwise_dist2 += np.triu(pariwise_dist2, k=1).T
    else:
        raise ValueError("No such kernel")
    return pariwise_dist2


def align(z_true: np.array, z_pred: np.array) -> np.array:
    """Align z_pred with z_true.

    Rotate, scale, shear, and translate z to make it aligned with z_true as best as possible.

    Parameters
    ----------
    z_true : ndarray of shape (n_points, d_latent)
        Ground truth of the latents, or points as references.
    z_pred : ndarray of shape (n_points, d_latent)
        Solved latents, or points to be aligned.

    Returns
    -------
    z_aligned : ndarray of shape (n_points, d_latent)
        Aligned version of z_pred.
    """

    n = z_pred.shape[0]
    z_aug = np.hstack((np.ones((n, 1)), z_pred))
    wtsaffine = np.linalg.lstsq(z_aug, z_true, rcond=None)[0] # affine transformation (rotate, scale, and translate)
    z_aligned = z_aug @ wtsaffine
    return z_aligned


def filt_cov_samp(cov_samp: np.array, variance=1) -> np.array:
    """Sample covariance matrix filter.

    Parameters
    ----------
    cov_samp : ndarray of shape (n_points, n_points)
        Sample covariance matrix.
    variance : float, optional
        Marginal variance, by default 1.

    Returns
    -------
    cov_samp_th : ndarray of shape (n_points, n_points)
        Filtered sample covariance matrix
    """

    cov_samp_th = cov_samp.copy()
    cov_samp_th[cov_samp_th <= 0] = 1e-5 # filter all negative covariance
    cov_samp_th[cov_samp_th > variance] = variance - 1e-5 # filter all > variance covariance, new added filter !!!
    np.fill_diagonal(cov_samp_th, variance)
    return cov_samp_th


def backward(x: np.array, method="sqrt") -> np.array:
    """Impute backward.

    Parameters
    ----------
    x : ndarray of shape (n_points, d_observation)
        Spikes or firing rates.
    method : str, optional
        ["log"|"sqrt"], by default "sqrt"

    Returns
    -------
    result : ndarray of shape (n_points, d_observation)
        backward(x).
    """

    x[x < 0] = 0
    if method == "log":
        tmp = np.log(x)
        for i in range(x.shape[1]):
            pos_idx = tmp[i] > 0
            neg_inf_idx = tmp[i] == -np.inf
            tmp[i, neg_inf_idx] = -np.mean(tmp[i, pos_idx]) / 0.81
        result = tmp
    elif method == "sqrt":
        result = np.sqrt(x)
    return result


def rigid_transform(x: np.array, y: np.array) -> tuple:
    """Rigid transformation.

    Rotation, reflection, and translation. Rigid transform x to y. Learn the rotation-reflection matrix r, and translation vector t. So that y = x r^T + 1 t^T, or y_i = r x_i + t.

    https://igl.ethz.ch/projects/ARAP/svd_rot.pdf
    http://nghiaho.com/?page_id=671

    Parameters
    ----------
    x : ndarray of shape (n_points, d)
        Dataset to be rigid transformed.
    y : ndarray of shape (n_points, d)
        Dataset to be aligned.

    Returns
    -------
    rh : ndarray of shape (d, d)
        Transpose of the rotation-reflection matrix R^H.
    t : ndarray of shape (d)
        Translation vector.
    """

    x_mean = np.mean(x, axis=0)
    y_mean = np.mean(y, axis=0)

    x_centered = x - x_mean
    y_centered = y - y_mean
    u, s, vh = np.linalg.svd(x_centered.T @ y_centered)
    rh = u @ vh
    t = y_mean - x_mean @ rh
    return rh, t


def rigid_align(z_true: np.array, z_pred: np.array) -> np.array:
    """Align z_pred with z_true using rigid transformation

    Rotate, scale, and translate z to make it aligned with z_true as best as possible.

    Parameters
    ----------
    z_true : ndarray of shape (n_points, d_latent)
        Ground truth of the latents, or points as references.
    z_pred : ndarray of shape (n_points, d_latent)
        Solved latents, or points to be aligned.

    Returns
    -------
    z_aligned : ndarray of shape (n_points, d_latent)
        Aligned version of z_pred.
    """

    rh, t = rigid_transform(z_pred, z_true)
    z_aligned = z_pred @ rh + t
    return z_aligned


def median_filter(z: np.array, kernel_size=3) -> np.array:
    """Median filter applied to latents, if you have a prior information that the latent is kind of smooth.

    Parameters
    ----------
    z : ndarray of shape (n_points, d_latent)
        Latents.
    kernel_size : int, optional
        Window/kernel size of the filter, by default 3.

    Returns
    -------
    z_medfilt: ndarray of shape (n_points, d_latent)
        Median filtered latents.
    """

    z_medfilt = np.zeros_like(z)
    for i in range(z.shape[1]):
        z_medfilt[:, i] = signal.medfilt(z[:, i], kernel_size=kernel_size)
    return z_medfilt