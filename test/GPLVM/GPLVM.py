import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
import logging
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
from sklearn.decomposition import PCA
from sklearn.manifold import SpectralEmbedding, Isomap
import GPy
import time
import argparse

from ikd import utils, core, evaluate, datasets


# fixed settings
d_latent = 3
n_points = 100
variance = 1
length_scale = 0.5
n_trials = 50


parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=int)
parser.add_argument('kernel_idx', type=int)
parser.add_argument('d_observation_idx', type=int)
parser.add_argument('method_idx', type=int)
args = parser.parse_args()
if args.kernel_idx == 0:
    kernel = "squared exponential"
    extra_kernel_hyperparam = None
    gplvm_kernel = GPy.kern.RBF(d_latent, variance=1, lengthscale=1)
elif args.kernel_idx == 1:
    kernel = "rational quadratic"
    extra_kernel_hyperparam = 1
    gplvm_kernel = GPy.kern.OU(d_latent, variance=1, lengthscale=1)
elif args.kernel_idx == 2:
    kernel = "gamma-exponential"
    extra_kernel_hyperparam = 1
    gplvm_kernel = GPy.kern.RatQuad(d_latent, variance=1, lengthscale=1, power=1)
elif args.kernel_idx == 3:
    kernel = "matern"
    extra_kernel_hyperparam = 1.5
    gplvm_kernel = GPy.kern.Matern32(d_latent, variance=1, lengthscale=1)
d_observation = (100, 200, 500, 1000, 2000, 5000, 10000)[args.d_observation_idx]
if args.method_idx == 0:
    method = 'PCA'
elif args.method_idx == 1:
    method = 'GPLVM'
elif args.method_idx == 2:
    method = 'IKD'


z_true = datasets.generate_latent(d_latent, args.dataset)
cov_true = utils.kernel_cov_generator(z_true, kernel=kernel, variance=variance, length_scale=length_scale, extra_kernel_hyperparam=extra_kernel_hyperparam)

df = pd.DataFrame(columns=['r2_true', 'mse_true', 'runtime', 'd_observation', 'trial', 'method', 'dataset', 'kernel'])

def learn_PCA(x):
    pca = PCA(n_components=d_latent)
    start = time.time()
    z_pca = pca.fit_transform(x)
    end = time.time()
    return z_pca, end-start

def learn_GPLVM(x):
    m_gplvm = GPy.models.GPLVM(x, d_latent, kernel=gplvm_kernel)
    m_gplvm.likelihood.variance = 1.
    start = time.time()
    m_gplvm.optimize(max_iters=1e4)
    end = time.time()
    z_gplvm = m_gplvm.X.values
    return z_gplvm, end-start

def learn_IKD(x):
    z_isomap = Isomap(n_components=d_latent).fit_transform(x)
    start = time.time()
    z_ikd = core.ikd_blockwise(x, d_latent, kernel=kernel, extra_kernel_hyperparam=extra_kernel_hyperparam, z_ref=z_isomap)
    end = time.time()
    return z_ikd, end-start


if method == 'PCA':
    learn = learn_PCA
elif method == 'GPLVM':
    learn = learn_GPLVM
elif method == 'IKD':
    learn = learn_IKD


for trial in range(n_trials):
    x = datasets.gaussian_process_generator(cov_true, d_observation=d_observation, seed=trial)

    np.random.seed(trial)
    z_pred, t = learn(x)
    z_pred_aligned = utils.align(z_true, z_pred)
    df.loc[trial] = [r2_score(z_true, z_pred_aligned), mean_squared_error(z_true, z_pred_aligned), t, d_observation, trial, method, args.dataset, kernel]

    logging.info(f"Trial {trial}")

df.to_csv(f'outputs/{args.dataset}_{kernel}_{d_observation}_{method}.csv')