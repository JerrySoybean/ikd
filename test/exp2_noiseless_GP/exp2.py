#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
import logging
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
from sklearn.decomposition import PCA
from sklearn.manifold import SpectralEmbedding

import argparse

from ikd import utils, core, evaluate, datasets

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=int)
parser.add_argument('kernel_idx', type=int)
parser.add_argument('d_observation_idx', type=int)
args = parser.parse_args()
dataset = args.dataset
kernel_idx = args.kernel_idx
d_observation_idx = args.d_observation_idx
kernel = ("squared exponential", "rational quadratic", "gamma-exponential", "matern")[kernel_idx]
extra_kernel_hyperparam = (None, 1, 1, 1.5)[kernel_idx]
d_observation = (200, 500, 1000, 2000, 5000, 10000, 20000)[d_observation_idx]

# fixed settings
d_latent = 3
n_points = 100
variance = 1
length_scale = 0.5
n_trials = 50

z_true = datasets.generate_latent(d_latent, dataset)
cov_true = utils.kernel_cov_generator(z_true, kernel=kernel, variance=variance, length_scale=length_scale, extra_kernel_hyperparam=extra_kernel_hyperparam)

df = pd.DataFrame(columns=['r2_true', 'r2_samp', 'mse_true', 'mse_samp', 'n_samples', 'trial', 'method', 'dataset', 'kernel'])

for trial in range(n_trials):
    x = datasets.gaussian_process_generator(cov_true, d_observation=d_observation, seed=trial)
    cov_samp_th = utils.filt_cov_samp(np.cov(x), variance=variance)
    pairwise_dist_from_samp = np.sqrt(utils.cov2dist2(cov_samp_th, kernel=kernel, variance=variance, length_scale=1, extra_kernel_hyperparam=extra_kernel_hyperparam))

    # PCA
    z_pca = PCA(n_components=d_latent).fit_transform(x)
    length_scale_pca = core.estimate_length_scale(pairwise_dist_from_samp, z_pca, cov_samp_th, variance=variance)
    cov_pca = utils.kernel_cov_generator(z_pca, kernel=kernel, variance=variance, length_scale=length_scale_pca, extra_kernel_hyperparam=extra_kernel_hyperparam)
    z_pca_aligned = utils.align(z_true, z_pca)
    df.loc[3*trial] = [r2_score(z_true, z_pca_aligned), r2_score(cov_samp_th.flatten(), cov_pca.flatten()), mean_squared_error(z_true, z_pca_aligned), mean_squared_error(cov_samp_th.flatten(), cov_pca.flatten()), d_observation, trial, 'pca', dataset, kernel]

    # IKD
    z_ikd = core.ikd(cov_samp_th, d_latent, kernel=kernel, extra_kernel_hyperparam=extra_kernel_hyperparam)[0]
    length_scale_ikd = core.estimate_length_scale(pairwise_dist_from_samp, z_ikd, cov_samp_th, variance=variance)
    cov_ikd = utils.kernel_cov_generator(z_ikd, kernel=kernel, variance=variance, length_scale=length_scale_ikd, extra_kernel_hyperparam=extra_kernel_hyperparam)
    z_ikd_aligned = utils.align(z_true, z_ikd)
    df.loc[3*trial+1] = [r2_score(z_true, z_ikd_aligned), r2_score(cov_samp_th.flatten(), cov_ikd.flatten()), mean_squared_error(z_true, z_ikd_aligned), mean_squared_error(cov_samp_th.flatten(), cov_ikd.flatten()), d_observation, trial, 'full', dataset, kernel]

    # IKD-b
    z_le = SpectralEmbedding(n_components=d_latent).fit_transform(x)
    z_ikd_b = core.ikd_blockwise(cov_samp_th, d_latent, kernel=kernel, extra_kernel_hyperparam=extra_kernel_hyperparam, clique_th_or_d_observation=d_observation, z_ref=z_le)
    length_scale_ikd_b = core.estimate_length_scale(pairwise_dist_from_samp, z_ikd_b, cov_samp_th, variance=variance)
    cov_ikd_b = utils.kernel_cov_generator(z_ikd_b, kernel=kernel, variance=variance, length_scale=length_scale_ikd_b, extra_kernel_hyperparam=extra_kernel_hyperparam)
    z_ikd_b_aligned = utils.align(z_true, z_ikd_b)
    df.loc[3*trial+2] = [r2_score(z_true, z_ikd_b_aligned), r2_score(cov_samp_th.flatten(), cov_ikd_b.flatten()), mean_squared_error(z_true, z_ikd_b_aligned), mean_squared_error(cov_samp_th.flatten(), cov_ikd_b.flatten()), d_observation, trial, 'block', dataset, kernel]

    logging.info(f"Trial {trial}")

df.to_csv(f'outputs/{dataset}_{kernel}_{d_observation}.csv')