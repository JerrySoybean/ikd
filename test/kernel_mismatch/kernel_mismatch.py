import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
import logging
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
from sklearn.decomposition import PCA
from sklearn.manifold import SpectralEmbedding, Isomap
import GPy
import argparse

from ikd import utils, core, evaluate, datasets


# fixed settings
d_latent = 3
dataset = 1
n_points = 100
variance = 1
length_scale = 0.5
n_trials = 50

kernel_list = ('squared exponential', 'rational quadratic', 'gamma-exponential', 'matern')
extra_kernel_hyperparam_list = (None, 1, 1, 1.5)

parser = argparse.ArgumentParser()
parser.add_argument('kernel_idx', type=int)
parser.add_argument('d_observation_idx', type=int)
parser.add_argument('method_idx', type=int)
args = parser.parse_args()
kernel = kernel_list[args.kernel_idx]
extra_kernel_hyperparam = extra_kernel_hyperparam_list[args.kernel_idx]
if args.kernel_idx == 0:
    gplvm_kernel = GPy.kern.RBF(d_latent, variance=1, lengthscale=1)
elif args.kernel_idx == 1:
    gplvm_kernel = GPy.kern.OU(d_latent, variance=1, lengthscale=1)
elif args.kernel_idx == 2:
    gplvm_kernel = GPy.kern.RatQuad(d_latent, variance=1, lengthscale=1, power=1)
elif args.kernel_idx == 3:
    gplvm_kernel = GPy.kern.Matern32(d_latent, variance=1, lengthscale=1)
d_observation = (100, 200, 500, 1000, 2000, 5000, 10000)[args.d_observation_idx]
if args.method_idx == 0:
    method = 'GPLVM'
elif args.method_idx == 1:
    method = 'IKD'

def learn_GPLVM(x):
    m_gplvm = GPy.models.GPLVM(x, d_latent, kernel=gplvm_kernel)
    m_gplvm.likelihood.variance = 1.
    m_gplvm.optimize(max_iters=1e4)
    z_gplvm = m_gplvm.X.values
    return z_gplvm

def learn_IKD(x):
    z_isomap = Isomap(n_components=d_latent).fit_transform(x)
    z_ikd = core.ikd_blockwise(x, d_latent, kernel=kernel, extra_kernel_hyperparam=extra_kernel_hyperparam, z_ref=z_isomap)
    return z_ikd

if method == 'GPLVM':
    learn = learn_GPLVM
elif method == 'IKD':
    learn = learn_IKD

z_true = datasets.generate_latent(d_latent, dataset)
for gen_kernel_idx in range(4):
    gen_kernel = kernel_list[gen_kernel_idx]
    extra_gen_kernel_hyperparam = extra_kernel_hyperparam_list[gen_kernel_idx]
    cov_true = utils.kernel_cov_generator(z_true, kernel=gen_kernel, variance=variance, length_scale=length_scale, extra_kernel_hyperparam=extra_gen_kernel_hyperparam)

    df = pd.DataFrame(columns=['$R^2$', 'MSE', 'd_observation', 'trial', 'method', 'gen_kernel', 'identify_kernel'])
    for trial in range(n_trials):
        x = datasets.gaussian_process_generator(cov_true, d_observation=d_observation, seed=trial)

        np.random.seed(trial)
        z_pred = learn(x)
        z_pred_aligned = utils.align(z_true, z_pred)
        df.loc[trial] = [r2_score(z_true, z_pred_aligned), mean_squared_error(z_true, z_pred_aligned), d_observation, trial, method, gen_kernel, kernel]

        logging.info(f"Trial {trial}")
    logging.info(f"Generating kernel {gen_kernel}")

    df.to_csv(f'outputs/{gen_kernel}_{kernel}_{d_observation}_{method}.csv')