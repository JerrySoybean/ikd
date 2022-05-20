#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import binned_statistic
from ikd.utils import kernel_cov_generator

def generate_latent(d_latent: int, dataset, n_points=100, seed=43, show=False) -> np.array:
    """Generate latent from pre-defined dataset
    
    Parameters
    ----------
    d_latent : int
        Latent dimensionality.
    dataset : int
        Index of the pre-defined dataset.
    d_latent : int, optional
        Number of points, by default 100.
    seed : int, optional
        Seed of random number generator, by default 43.
    show : bool, optional
        Visualization flag, by default False.

    Returns
    -------
    z_true : ndarray of shape (d_latent, d_latent)
        Ground truth of the latent variables.
    """
    rng = np.random.default_rng(seed=seed)

    if d_latent == 1:
        if dataset == 0:
            # sin curve
            z_true = rng.uniform(low=0, high=10, size=n_points)
            z_true = z_true[:, np.newaxis]
        elif dataset == 1:
            # neuron data
            z_true = loadmat(__file__[:-11] + 'data/simdata1.mat')['simdata'][0, 0][0]
        else:
            raise ValueError("No such dataset")

        if show:
            plt.figure()
            plt.plot(z_true[:, 0])
            plt.xlabel('$t$')
            plt.ylabel('$z$')
            plt.title("True latents: $z_\\mathrm{true}$")

    if d_latent == 2:
        if dataset == 0:
            # sin curve
            b1 = np.linspace(2.2, 5, n_points)
            noise = 0.05
            b2 = np.sin(b1 * 8) + rng.normal(scale=noise, size=n_points)
            b3 = np.cos(b1 * 2) + rng.normal(scale=noise, size=n_points)
            z_true = np.array([b3, b2]).T
        elif dataset == 1:
            from matplotlib import image
            img = image.imread(__file__[:-11] + 'data/s_curve_demo.png')[:, :, 1]
            row, column = np.where(img < 0.98)
            ii = rng.choice(len(row), size=n_points, replace=False) # choose n from len(row)
            noise = 0.05
            z_true = np.array([column, row])[:, ii].T / 100 + rng.normal(scale=noise, size=(n_points, 2))
        elif dataset == 2:
            b1 = np.linspace(2.2, 5, n_points)
            b2 = np.sin(b1 * 3)
            b3 = np.cos(b1 * 36)
            z_true = np.array([b3, b2]).T
        elif dataset == 3:
            b1 = rng.normal(size=n_points)
            b2 = np.sin(b1 * 3) + rng.normal(size=n_points)
            z_true = np.array([b1, b2]).T
        elif dataset == 5:
            # sin curve
            b1 = np.linspace(2.2, 5, n_points)
            noise = 0.3
            b2 = np.sin(b1*8) + rng.normal(scale=noise, size=n_points)
            z_true = np.array([b1, b2]).T
        else:
            raise ValueError("No such dataset")
        if show:
            plt.figure()
            plt.plot(z_true[:, 0], z_true[:, 1])
            plt.xlabel('$z_1$')
            plt.ylabel('$z_2$')
            plt.title("True latents: $z_\\mathrm{true}$")
        
    elif d_latent == 3:
        if dataset == 0:
            theta = np.linspace(-4 * np.pi, 4 * np.pi, n_points)
            z = np.concatenate((np.linspace(-2, 1.95, 50) / 4, np.linspace(2, -2, 50) / 4))
            r = (z)**3 + 1
            noise = 0.05
            x = 0.5 * r * np.sin(theta) + rng.normal(scale=noise, size=n_points)
            y = 0.5 * r * np.cos(theta) + rng.normal(scale=noise, size=n_points)
            z_true = 2 * (np.array([x, y, z]).T)[np.concatenate((np.arange(0, 25), np.arange(74, 49, -1), np.arange(25, 50), np.arange(99, 74, -1)))]
        elif dataset == 1:
            theta = np.linspace(-4 * np.pi, 4 * np.pi, n_points)
            z = np.linspace(-2, 2, n_points) / 4
            r = (z)**2 + 1
            x = r * np.sin(1.3 * theta)
            y = r * np.cos(theta)
            z_true = np.array([x, y, z]).T
        elif dataset == 2:
            n_points = 100
            t = np.linspace(-0.5, 0.5, 10)
            x, y = np.meshgrid(t, t)
            z = np.sin(2 * np.pi * x) + np.cos(3 * np.pi * y) / 2
            idx = rng.permutation(n_points)
            if show:
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                ax.plot_surface(x, y, z)
                ax.set_xlabel('$z_1$')
                ax.set_ylabel('$z_2$')
                ax.set_zlabel('$z_3$')
            z_true = np.array([x.flatten()[idx], y.flatten()[idx], z.flatten()[idx]]).T
        else:
            raise ValueError("No such dataset")
        if show:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            plt.plot(z_true[:, 0], z_true[:, 1], z_true[:, 2])
            ax.set_xlabel('$z_1$')
            ax.set_ylabel('$z_2$')
            ax.set_zlabel('$z_3$')
            plt.title("True latents: $z_\\mathrm{true}$")
    return z_true


def hc_6(mouse_name: str, day: int, epoch: int, standardize_pos=False, show=False):
    """Hippocampus dataset 6.

    http://crcns.org/data-sets/hc/hc-6/about-hc-5

    Parameters
    ----------
    mouse_name : str
        Mouse name.
    day : int
        The day-th day.
    epoch : int
        The epoch-th epoch.
    standardize_pos : bool, optional
        If standardize the position within [0, 1] x [0, 1], by default True.
    show : bool, optional
        Visualization flag, by default False.

    Returns
    -------
    position : ndarray of shape (n_points, 2)
        Ground truth of the latent variables. For the hc_6 dataset, it is just the (x, y) location of the mouse in a W shaped maze.
    spikes : ndarray of shape (n_points, n_neurons)
        Recorded spike train.
    """

    if loadmat(__file__[:-11] + f'data/hc-6/{mouse_name}/{mouse_name}task{day:02d}.mat', squeeze_me=False, struct_as_record=False)['task'][0, day-1][0, epoch-1][0, 0].type[0] != 'run':
        raise ValueError('It is not a "run" epoch')

    tetrode_list = []
    cell_list = []
    cellinfo_day_epoch = loadmat(__file__[:-11] + f'data/hc-6/{mouse_name}/{mouse_name}cellinfo.mat', squeeze_me=False, struct_as_record=False)['cellinfo'][0, day-1][0, epoch-1]

    for tetrode in range(cellinfo_day_epoch.shape[1]):
        cellinfo_day_epoch_tetrode = cellinfo_day_epoch[0, tetrode]
        n_cells = cellinfo_day_epoch_tetrode.shape[1]
        if n_cells != 0:
            for cell in range(n_cells):
                cellinfo_day_epoch_tetrode_cell = cellinfo_day_epoch_tetrode[0, cell]
                try:
                    if cellinfo_day_epoch_tetrode_cell[0, 0].area[0] in ['CA1', 'CA3', 'MEC']:
                        tetrode_list.append(tetrode)
                        cell_list.append(cell)
                except:
                    pass

    pos = loadmat(__file__[:-11] + f'data/hc-6/{mouse_name}/{mouse_name}pos{day:02d}.mat', squeeze_me=False, struct_as_record=False)['pos'][0, day-1][0, epoch-1][0, 0]

    spikes = loadmat(__file__[:-11] + f'data/hc-6/{mouse_name}/{mouse_name}spikes{day:02d}.mat', squeeze_me=False, struct_as_record=False)['spikes'][0, day-1][0, epoch-1]

    scale = 10
    t_start = np.floor(pos.data[0, 0] * scale) / scale
    time_bins = np.arange(t_start, pos.data[-1, 0] + 1, 1)

    position = binned_statistic(pos.data[:, 0], pos.data[:, 1:3].T, bins=time_bins)[0].T
    if standardize_pos is True:
        pos_min = np.min(position, axis=0)
        pos_max = np.max(position, axis=0)
        position = (position - pos_min) / (pos_max - pos_min)

    result = []
    for tetrode, cell in zip(tetrode_list, cell_list):
        curr_spikes = spikes[0, tetrode][0, cell][0, 0].data
        if len(curr_spikes) != 0:
            result.append(np.histogram(curr_spikes[:, 0], bins=time_bins)[0])
    spikes = np.array(result, dtype=float).T

    # try:
    #     curr_data_file = loadmat(__file__[:-11] + f"data/hc-6/neuron_{mouse_name}_d{day}_e{epoch}_s1.mat")
    #     z_true = curr_data_file["xx"]
    #     spikes = curr_data_file["yy"]
    # except:
    #     raise ValueError("No such dataset.")
    # s = 2
    # while True:
    #     try:
    #         curr_data_file = loadmat(__file__[:-11] + f"data/hc-6/neuron_{mouse_name}_d{day}_e{epoch}_s{s}.mat")
    #         z_true = np.vstack((z_true, curr_data_file["xx"]))
    #         spikes = np.vstack((spikes, curr_data_file["yy"]))
    #         s += 1
    #     except:
    #         break
    if show:
        plt.figure()
        plt.plot(position[:, 0], position[:, 1])
        plt.xlabel('$z_1$')
        plt.ylabel('$z_2$')
        plt.title("True latents: $z_\\mathrm{true}$, which is the mouse's location")
    return position, spikes


def generate_latent_from_prior(d_latent: int, n_points=100, kernel='autoregressive', variance=1, length_scale=1, bound=None, seed=43, show=False) -> np.array:
    """Generate latent from a specified prior.
    
    Parameters
    ----------
    d_latent : int
        Latent dimensionality.
    dataset : int
        Index of the pre-defined dataset.
    n_points : int, optional
        Number of latent points, by default 100.
    kernel : str, optional
        ["autoregressive" | "squared exponential"], by default "autoregressive".
    variance : int, optional
        Marginal variance, by default 1.
    length_scale : int, optional
        Length scale of the kernel, by default 1.
    bound : int, optional
        Bound for latent, by default None. If is is provided, any value in latent exceeds this bound will be set to this bound.
    seed : int, optional
        Seed of random number generator, by default 43.
    show : bool, optional
        Visualization flag, by default False.

    Returns
    -------
    z_true : ndarray of shape (n_points, d_latent)
        Ground truth of the latent variables.
    """

    rng = np.random.default_rng(seed=seed)
    z_true = rng.multivariate_normal(mean=np.zeros(n_points), cov=kernel_cov_generator(np.arange(n_points)[:, np.newaxis], kernel=kernel, variance=variance, length_scale=length_scale), size=d_latent).T
    if bound is not None:
        z_true[z_true > bound] = bound
        z_true[z_true < -bound] = -bound
    return z_true


def gaussian_process_generator(cov_true: np.array, d_observation: int, seed=43) -> np.array:
    """Generate several random variables from Gaussian process.

    Parameters
    ----------
    cov_true : ndarray of shape (n_points, n_points)
        Kernel covariance matrix.
    d_observation : int
        Observation dimensionality, equal to the number of Gaussian samples.
    seed : int, optional
        Seed of the random number generator, by default 43.

    Returns
    -------
    x : ndarray of shape (n_points, d_observation)
        Samples from GP under cov_true.
    """

    rng = np.random.default_rng(seed=seed)
    x = rng.multivariate_normal(mean=np.zeros(cov_true.shape[0]), cov=cov_true, size=d_observation).T
    return x


def gaussian_bump_generator(z_true: np.array, n_grids_per_dim: int, n_centers: int, area: int, variance=1, length_scale=1, seed=43) -> np.array:
    """Generate Gaussian bump data.

    Parameters
    ----------
    z_true : ndarray of shape (n_points, d_latent)
        Ground truth of the latent variables.
    n_grids_per_dim : int
        Number of candidate centers (grids) per dimension d.
    n_centers : int
        Number of Gaussian bump centers, a.k.a. the dimension of the output data x. It should be less than or equal to n_grids^2.
    area : int
        Range of the Gaussian bump, [-area, area]^d_latent.
    variance : int, optional
        Marginal variance, by default 1.
    length_scale : int, optional
        Length scale of the kernel, by default 1.
    seed : int, optional
        Seed of random number generator, by default 43.

    Returns
    -------
    x : nd array of shape (n_points, n_centers)
        Observation data.
    """

    d = z_true.shape[1]
    if n_grids_per_dim**d < n_centers:
        raise ValueError("Number of Gaussian bump centers exceeds the number of its candidate.")
    rng = np.random.default_rng(seed=seed)
    step = 2 * area / (n_grids_per_dim - 1)
    grid = np.mgrid[tuple(slice(-area, area + step, step) for _ in range(d))]
    bump_centers = np.array(list(map(np.ravel, grid))).T
    idx = rng.permutation(bump_centers.shape[0])[:n_centers]
    bump_centers = bump_centers[idx]
    x = kernel_cov_generator(z_true, bump_centers, kernel='autoregressive', variance=variance, length_scale=length_scale)
    return x
