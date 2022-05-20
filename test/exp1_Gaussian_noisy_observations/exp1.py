#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
import logging
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import SpectralEmbedding

import argparse

from ikd import utils, core, evaluate, datasets

parser = argparse.ArgumentParser()
parser.add_argument('f_idx', type=int)
parser.add_argument('d_observation_idx', type=int)
args = parser.parse_args()
d_observation_idx = args.d_observation_idx
f_idx = args.f_idx
d_observation = (10, 20, 50, 100, 200, 500, 1000)[d_observation_idx]
f = ('sin', 'Gaussian bump', 'GP')[f_idx]

# fixed settings
n_points = 1000
n_trials = 50

df = pd.DataFrame(columns=['r2', 'mse', 'f', 'n_attributes', 'trial', 'method'])

for trial in range(n_trials):
    rng = np.random.default_rng(seed=trial)
    if f == 'sin':
        d = 1
        z_true = datasets.generate_latent_from_prior(d, n_points, kernel='autoregressive', variance=6, length_scale=5, bound=6, seed=trial)
        omega = rng.uniform(low=-1, high=1, size=(d_observation, d))
        phi = rng.uniform(low=-np.pi, high=np.pi, size=(1, d_observation))
        x = np.sin(z_true @ omega.T + phi) + rng.normal(scale=0.1, size=(n_points, d_observation))
    elif f == 'Gaussian bump':
        d = 2
        z_true = datasets.generate_latent_from_prior(d, n_points, kernel='autoregressive', variance=6, length_scale=5, bound=6, seed=trial)
        x = datasets.gaussian_bump_generator(z_true, 100, d_observation, area=6, variance=20, length_scale=1, seed=trial) + rng.normal(scale=0.05, size=(n_points, d_observation))
    elif f == 'GP':
        d = 3
        z_true = datasets.generate_latent_from_prior(d, n_points, kernel='autoregressive', variance=6, length_scale=5, bound=6, seed=trial)
        cov_true = utils.kernel_cov_generator(z_true, length_scale=3)
        x = datasets.gaussian_process_generator(cov_true, d_observation, seed=trial) + rng.normal(scale=0.05, size=(n_points, d_observation))
    cov_samp = np.cov(x)
    if f == 'GP':
        variance_samp = np.mean(cov_samp) * 3.5
    else:
        variance_samp = np.mean(cov_samp) * 2
    cov_samp_th = utils.filt_cov_samp(cov_samp, variance_samp)

    # PCA
    z_pca = PCA(n_components=d).fit_transform(x)
    z_pca_aligned = utils.align(z_true, z_pca)
    df.loc[5*trial] = [r2_score(z_true, z_pca_aligned), mean_squared_error(z_true, z_pca_aligned), f, d_observation, trial, 'PCA']

    # kernel PCA
    if f == 'sin':
        z_kernel_pca = KernelPCA(n_components=d, kernel='cosine').fit_transform(x)
    else:
        z_kernel_pca = KernelPCA(n_components=d, kernel='sigmoid').fit_transform(x)
    z_kernel_pca_aligned = utils.align(z_true, z_kernel_pca)
    df.loc[5*trial+1] = [r2_score(z_true, z_kernel_pca_aligned), mean_squared_error(z_true, z_kernel_pca_aligned), f, d_observation, trial, 'kernel PCA']

    # Laplacian eigenmaps
    z_le = SpectralEmbedding(n_components=d).fit_transform(x)
    z_le_aligned = utils.align(z_true, z_le)
    df.loc[5*trial+2] = [r2_score(z_true, z_le_aligned), mean_squared_error(z_true, z_le_aligned), f, d_observation, trial, 'Laplacian eigenmaps']

    # IKD
    z_ikd = core.ikd(cov_samp_th, d, variance=variance_samp)[0]
    z_ikd_aligned = utils.align(z_true, z_ikd)
    df.loc[3*trial+3] = [r2_score(z_true, z_ikd_aligned), mean_squared_error(z_true, z_ikd_aligned), f, d_observation, trial, 'IKD']

    # blockwise
    if f == 'sin':
        z_ikd_b = core.ikd_blockwise(cov_samp_th, d, variance=variance_samp, clique_th_or_d_observation=0.3, z_ref=z_le)
    else:
        z_ikd_b = core.ikd_blockwise(cov_samp_th, d, variance=variance_samp, clique_th_or_d_observation=d_observation, z_ref=z_le)
    z_ikd_b_aligned = utils.align(z_true, z_ikd_b)
    df.loc[3*trial+4] = [r2_score(z_true, z_ikd_b_aligned), mean_squared_error(z_true, z_ikd_b_aligned), f, d_observation, trial, 'IKD-b']

    logging.info(f"Trial {trial}")

df.to_csv(f'outputs/{f}_{d_observation}.csv')