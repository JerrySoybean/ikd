import numpy as np
import pandas as pd
import argparse
import logging

from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import SpectralEmbedding

from sklearn.metrics import r2_score
from ikd import utils, core, datasets, epca

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('d_observation', type=int)
parser.add_argument('backward', type=str, choices=["sqrt", "log"])

args = parser.parse_args()
d_observation = args.d_observation

n_trials = 50
result = pd.DataFrame(np.zeros((n_trials, 9)), columns=["PCA", 'kernel PCA', 'Laplacian eigenmaps', "epca_collin", "$e$PCA", "IKD", "IKD-b", "$e$IKD", "$e$IKD-b"])

for trial in range(n_trials):
    rng = np.random.default_rng(trial)
    n_points, d_latent = 1000, 1
    z_true = datasets.generate_latent_from_prior(1, 1000, kernel='autoregressive', variance=9, length_scale=100, bound=6, seed=trial)
    kernel = "squared exponential"
    variance = 2
    length_scale = 1
    cov_true = utils.kernel_cov_generator(z_true, kernel=kernel, variance=variance, length_scale=length_scale)
    x = rng.multivariate_normal(mean=np.zeros(1000), cov=cov_true, size=d_observation).T
    firing_rates = np.exp(x)
    spikes = rng.poisson(firing_rates).astype(float)
    
    spikes_dense = utils.backward(spikes, method=args.backward)
    cov_spikes = np.cov(spikes_dense)
    variance_spikes = 2 * np.mean(cov_spikes)
    cov_spikes_th = utils.filt_cov_samp(cov_spikes, variance_spikes)

    z_pca = PCA(n_components=d_latent).fit_transform(spikes_dense)
    z_kernel_pca = KernelPCA(n_components=d_latent, kernel='sigmoid').fit_transform(spikes_dense)
    z_le = SpectralEmbedding(n_components=d_latent).fit_transform(spikes_dense)
    z_epca_collin = epca.poisson_pca(spikes, n_components=d_latent)[0]
    z_epca_liu = epca.exp_fam_pca(spikes, "poisson", r=d_latent)[3]

    z_ikd = core.ikd(cov_spikes_th, d_latent, kernel=kernel, variance=variance_spikes, length_scale=1)[0]
    z_ikd_b = core.ikd_blockwise(cov_spikes_th, d_latent, kernel=kernel, variance=variance_spikes, length_scale=1, clique_th_or_d_observation=d_observation, z_ref=z_le)

    sol = epca.exp_fam_pca(spikes.T, "poisson", r=d_observation)
    firing_rates_est = epca.wiener_filter(sol[0], spikes.T, "poisson").T
    firing_rates_est[firing_rates_est < 0] = 0

    x_est = utils.backward(firing_rates_est, method=args.backward)
    cov_est = np.cov(x_est)
    variance_est = 2 * np.mean(cov_est)
    cov_est_th = utils.filt_cov_samp(cov_est, variance_est)
    z_eikd = core.ikd(cov_est_th, d_latent, kernel=kernel, variance=variance_spikes, length_scale=1)[0]
    z_eikd_b = core.ikd_blockwise(cov_est_th, d_latent, kernel=kernel, variance=variance_est, length_scale=1, clique_th_or_d_observation=d_observation, z_ref=z_le)

    result.iloc[trial] = list(map(lambda x: r2_score(z_true, utils.align(z_true, x)), [z_pca, z_kernel_pca, z_le, z_epca_collin, z_epca_liu, z_ikd, z_ikd_b, z_eikd, z_eikd_b]))

    logging.info(f"Trial {trial}")
result.to_csv(f"outputs/{d_observation}_{args.backward}.csv")