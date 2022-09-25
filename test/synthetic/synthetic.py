#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
import logging
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import SpectralEmbedding, LocallyLinearEmbedding, Isomap, TSNE
import umap
import GPy
import time
import argparse

from ikd import utils, core, evaluate, datasets


# fixed settings
n_points = 1000
n_trials = 50


parser = argparse.ArgumentParser()
parser.add_argument('f_idx', type=int)
parser.add_argument('d_observation_idx', type=int)
parser.add_argument('method_idx', type=int)
args = parser.parse_args()
d_observation_idx = args.d_observation_idx
f_idx = args.f_idx
d_observation = (10, 20, 50, 100, 200, 500, 1000)[d_observation_idx]
f = ('sin', 'Gaussian bump', 'GP')[f_idx]

if f == 'sin':
    d_latent = 1
elif f == 'Gaussian bump':
    d_latent = 2
elif f == 'GP':
    d_latent = 3

def learn_PCA(x):
    pca = PCA(n_components=d_latent)
    start = time.time()
    z_pca = pca.fit_transform(x)
    end = time.time()
    return z_pca, end-start

def learn_KPCA(x):
    if f == 'sin':
        kpca = KernelPCA(n_components=d_latent, kernel='cosine')
    else:
        kpca = KernelPCA(n_components=d_latent, kernel='sigmoid')
    start = time.time()
    z_kpca = kpca.fit_transform(x)
    end = time.time()
    return z_kpca, end-start

def learn_LE(x):
    le = SpectralEmbedding(n_components=d_latent)
    start = time.time()
    z_le = le.fit_transform(x)
    end = time.time()
    return z_le, end-start

def learn_LLE(x):
    lle = LocallyLinearEmbedding(n_components=d_latent)
    start = time.time()
    z_lle = lle.fit_transform(x)
    end = time.time()
    return z_lle, end-start

def learn_TSNE(x):
    tsne = TSNE(n_components=d_latent, init='pca', learning_rate='auto', random_state=42)
    start = time.time()
    z_tsne = tsne.fit_transform(x)
    end = time.time()
    return z_tsne, end-start

def learn_Isomap(x):
    isomap = Isomap(n_components=d_latent)
    start = time.time()
    z_isomap = isomap.fit_transform(x)
    end = time.time()
    return z_isomap, end-start

def learn_UMAP(x):
    uma1 = umap.UMAP(n_components=d_latent, random_state=42)
    start = time.time()
    z_umap = uma1.fit_transform(x)
    end = time.time()
    return z_umap, end-start

def learn_GPLVM(x):
    m_gplvm = GPy.models.GPLVM(x, d_latent, kernel=GPy.kern.RBF(d_latent, variance=1, lengthscale=1))
    m_gplvm.likelihood.variance = 1.
    start = time.time()
    m_gplvm.optimize(max_iters=1e3)
    end = time.time()
    z_gplvm = m_gplvm.X.values
    return z_gplvm, end-start

def learn_IKD(x):
    start = time.time()
    if f == 'Gaussian bump':
        z_ikd = core.ikd_blockwise(x, d_latent, clique_th=0.2, max_n_cliques=2)
    elif f == 'sin':
        z_ikd = core.ikd_blockwise(x, d_latent, clique_th=0.6, max_n_cliques=2)
    elif f == 'GP':
        z_ikd = core.ikd_blockwise(x, d_latent, clique_th=0.6, max_n_cliques=2)
    end = time.time()
    return z_ikd, end-start

if args.method_idx == 0:
    method = 'PCA'
    learn = learn_PCA
elif args.method_idx == 1:
    method = 'KPCA'
    learn = learn_KPCA
elif args.method_idx == 2:
    method = 'LE'
    learn = learn_LE
elif args.method_idx == 3:
    method = 'LLE'
    learn = learn_LLE
elif args.method_idx == 4:
    method = 'TSNE'
    learn = learn_TSNE
elif args.method_idx == 5:
    method = 'Isomap'
    learn = learn_Isomap
elif args.method_idx == 6:
    method = 'UMAP'
    learn = learn_UMAP
elif args.method_idx == 7:
    method = 'GPLVM'
    learn = learn_GPLVM
elif args.method_idx == 8:
    method = 'IKD'
    learn = learn_IKD


df = pd.DataFrame(columns=['$R^2$', 'MSE', 'runtime', 'f', 'd_observation', 'trial', 'method'])

for trial in range(n_trials):
    rng = np.random.default_rng(seed=trial)
    z_true = datasets.generate_latent_from_prior(d_latent, n_points, kernel='autoregressive', variance=6, length_scale=5, bound=6, seed=trial)
    if f == 'sin':
        omega = rng.uniform(low=-1, high=1, size=(d_observation, d_latent))
        phi = rng.uniform(low=-np.pi, high=np.pi, size=(1, d_observation))
        x = np.sin(z_true @ omega.T + phi) + rng.normal(scale=0.1, size=(n_points, d_observation))
    elif f == 'Gaussian bump':
        x = datasets.gaussian_bump_generator(z_true, 100, d_observation, area=6, variance=20, length_scale=1, seed=trial) + rng.normal(scale=0.05, size=(n_points, d_observation))
    elif f == 'GP':
        cov_true = utils.kernel_cov_generator(z_true, length_scale=3)
        x = datasets.gaussian_process_generator(cov_true, d_observation, seed=trial) + rng.normal(scale=0.05, size=(n_points, d_observation))

    np.random.seed(trial)
    z_pred, t = learn(x)
    z_pred_aligned = utils.align(z_true, z_pred)
    df.loc[trial] = [r2_score(z_true, z_pred_aligned), mean_squared_error(z_true, z_pred_aligned), t, f, d_observation, trial, method]
    logging.info(f"Trial {trial}")

df.to_csv(f'outputs/{f}_{d_observation}_{method}.csv')