#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

def compare_latents(z_true: np.array, z_pred: np.array, name: str, measurement="r2") -> None:
    """Compare estimated latents with ground truth.

    Parameters
    ----------
    z_true : ndarray of shape (n_points, d_latent)
        Ground truth of the latents.
    z_pred : ndarray of shape (n_points, d_latent)
        Estimated latents.
    name : str
        Title of the plots, usually be the method name.
    measurement : str, optional
        "r2" (coefficient of determinant) or "mse" (MSE, mean squared error), by default "r2".
    """
    
    d_latent = z_pred.shape[1]
    if d_latent == 1:
        plt.figure(figsize=(10, 4))
        plt.plot(z_true, label='true')
        plt.plot(z_pred, label=name)
        plt.legend()
        plt.xlabel('$n$ points')
        plt.ylabel('$z$')
    
    elif d_latent == 2:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(z_true[:, 0], z_true[:, 1], label='true')
        plt.plot(z_pred[:, 0], z_pred[:, 1], label=name)
        plt.legend()
        plt.xlabel('$z_1$')
        plt.ylabel('$z_2$')

        plt.subplot(2, 2, 2)
        plt.plot(z_true[:, 0])
        plt.plot(z_pred[:, 0])
        plt.xlabel('$z_1$')

        plt.subplot(2, 2, 4)
        plt.plot(z_true[:, 1])
        plt.plot(z_pred[:, 1])
        plt.xlabel('$z_2$')
    
    elif d_latent == 3:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        plt.plot(z_true[:, 0], z_true[:, 1], z_true[:, 2], label='true')
        plt.plot(z_pred[:, 0], z_pred[:, 1], z_pred[:, 2], label=name)
        plt.legend()
        ax.set_xlabel('$z_1$')
        ax.set_ylabel('$z_2$')
        ax.set_zlabel('$z_3$')

        plt.subplot(3, 2, 2)
        plt.plot(z_true[:, 0])
        plt.plot(z_pred[:, 0])
        plt.xlabel('$z_1$')

        plt.subplot(3, 2, 4)
        plt.plot(z_true[:, 1])
        plt.plot(z_pred[:, 1])
        plt.xlabel('$z_2$')

        plt.subplot(3, 2, 6)
        plt.plot(z_true[:, 2])
        plt.plot(z_pred[:, 2])
        plt.xlabel('$z_3$')
    
    else:
        fig = plt.figure(figsize=(6, d_latent*2))
        for i in range(d_latent):
            plt.subplot(d_latent, 1, i+1)
            plt.plot(z_true[:, i])
            plt.plot(z_pred[:, i])
            plt.xlabel(f'$z_{i+1}$')

    if measurement == "r2":
        plt.suptitle(f'{name}\n$R^2$ = {r2_score(z_true, z_pred):.4f}')
    elif measurement == "mse":
        plt.suptitle(f"{name}\nMSE = {mean_squared_error(z_true, z_pred):.4f}")
    
    plt.tight_layout()


def compare_cov(cov_samp_th: np.array, cov_from_z_pred_raw: np.array, cov_true=None, measurement="r2") -> None:
    """Compare covariance matrix.

    Parameters
    ----------
    cov_samp_th : ndarray of shape (n_points, n_points)
        Sample covariance matrix
    cov_from_z_pred_raw : ndarray of shape (n_points, n_points)
        Covariance matrix generated from unaligned z_pred.
    cov_true : ndarray of shape (n_points, n_points), optional
        Ground truth of the kernel covariance matrix, by default None
    measurement : str, optional
        "r2" (coefficient of determinant) or "mse" (MSE, mean squared error), by default "r2"
    """

    if cov_true is None:
        fig, axs = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    else:
        fig, axs = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    axs0 = axs[0].matshow(cov_samp_th, vmin=0, vmax=1)
    axs[0].set_title("filtered sample covariance matrix")

    axs[1].matshow(cov_from_z_pred_raw, vmin=0, vmax=1)
    axs[1].set_title("recovered covariance matrix")

    if cov_true is None:
        fig.colorbar(axs0, ax=axs)
    else:
        axs[2].matshow(cov_true, vmin=0, vmax=1)
        axs[2].set_title("true covariance matrix")
        fig.colorbar(axs0, ax=axs)
    if measurement == "r2":
        plt.suptitle(f"$R^2$ = {r2_score(cov_samp_th.flatten(), cov_from_z_pred_raw.flatten()):.4f}")
    elif measurement == "mse":
        plt.suptitle(f"MSE = {mean_squared_error(cov_samp_th.flatten(), cov_from_z_pred_raw.flatten()):.4f}")
