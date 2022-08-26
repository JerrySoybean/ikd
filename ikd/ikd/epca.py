#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import minimize

'''
Adapted from https://github.com/lydiatliu/epca

In this MATLAB package and the corresponding paper, the authors provide epca (exponential family PCA) implementation, and some other application tools. They claim their epca algorithm with analytical solution is outperforms Collins' iterative EPCA algorithm.
'''

def standard_spiked_forward(ell: np.array, gamma: float):
    '''
    Downloaded from /dobriban/EigenEdge/master/Code/standard_spiked_forward.m
    Part of Edgar Dobriban's EigenEdge software package
    see https://github.com/dobriban/EigenEdge

    Compute the characteristics of the standard spiked model
    Using explicit formulas

    Parameters
    ell : population spike location (null: ell=0, so that eigenvalue of population covariance matrix is 1+ell)
    gamma : aspect ratio p/n

    Returns
    lam : asymptotic sample spike location (eigenvalue)
    cos_right, cos_left : squares of the right/left singular value angles
    m, v : Stieltjes transform and companion ST
    '''

    k = len(ell)
    lam = np.zeros(k)
    cos_right = np.zeros(k)
    cos_left = np.zeros(k)
    v = np.zeros(k)
    gamma_minus = (1 - np.sqrt(gamma))**2
    gamma_plus = (1 + np.sqrt(gamma))**2

    for i in range(k):
        if np.abs(ell[i]) < gamma**0.5: # BBP PT
            lam[i] = (1 + gamma**0.5)**2
            cos_right[i] = 0
            cos_left[i] = 0
        else:
            lam[i] = (1 + ell[i]) * (1 + gamma/ell[i])
            cos_right[i] = (1 - gamma/ell[i]**2) / (1 + gamma/ell[i])
            cos_left[i] = (1 - gamma/ell[i]**2) / (1 + 1/ell[i])
        
        x = lam[i]
        im_mult = 1
        if (x > gamma_minus) and (x < gamma_plus):
            im_mult = 1j
        v[i] = 1 / (2*x) * (-(1+x-gamma) + im_mult*(np.abs((1+x-gamma)**2 - 4*x))**0.5)
    m = 1/gamma*v - (1 - 1/gamma) / lam
    return lam, cos_right, cos_left, m, v


def op_norm_shrink2(eig, gamma):
    '''
    operator norm shrinkage
    from page 13 (4.4) of Donoho et al 2013 "Optimal Shrinkage..."
    available at https://arxiv.org/abs/1311.0851
    '''

    l = len(eig)
    shr_eig = np.zeros(l)
    for i in range(l):
        if eig[i] > (1 + np.sqrt(gamma))**2:
            shr_eig[i] = (eig[i] + 1 - gamma + np.sqrt((eig[i] + 1 - gamma)**2 - 4 * eig[i])) / 2
        else:
            shr_eig[i] = 1 + gamma**0.5
    return shr_eig


def exp_fam_pca(Y, exp_fam_type="normal", r=1, sigma_sq=1, bino_n=1):
    """ePCA

    High Dimensional Exponential Family PCA
    computes a Principal Component decomposition of exponential-family valued data
    see the ePCA paper by Liu, Dobriban and Singer for details of the method

    Parameters
    ----------
    Y : ndarray of shape (n, p)
        nxp data matrix with n rows containing p-dimensional noisy signal vectors.
    exp_fam_type : str, optional
        Distribution of the entries of Y, a type of exponential family, ['normal'|'poisson'|'binomial'], by default "normal".
    r : int, optional
        Rank estimate used in PCA and denoising, by default 1.
    sigma_sq : int, optional
        Parameters for Gaussian distribution, by default 1.
    bino_n : int, optional
        Parameters for binomial distribution, by default 1.

    Returns
    -------
    S_recolored : ndarray of shape (p, p)
        Whitened, shrunk and recolored covariance matrix.
    white_shr_eval : ndarray of shape (r,)
        Whitened, shrunk eigenvalue (no recoloring)
    white_V : ndarray of shape (r,)
        Top r eigenvectors of whitened covariance matrix (also right singular vectors of centered and whitened data matrix)
    white_U : ndarray of shape (r,)
        Top r left singular vectors of centered and whitened data matrix.
    white_eval : ndarray of shape (r,)
        Whitened eigenvalues (no shrinkage, or recoloring) (standard Marchenko-Pastur)
    recolor_eval : ndarray of shape (p,)
        Eigenvalues of S_w_op
    recolor_V : ndarray of shape (p, p)
        Eigenvectors of S_w_op
    D_n : ndarray of shape (p, p)
        Diagonal debiasing matrix.
    white_shr_eval_scaled : float
        Scaling to white_shr_eval, to remove bias for estimating true spike.
    estim_SNR_improvement : float
        Estimated SNR improvement due to whitening.
    """

    n, p = Y.shape
    gamma = p/n

    # impute missing data to column means
    impute_missing = True
    if impute_missing is True:
        m = np.nanmean(Y, axis=0)
        for i in range(p):
            ind_nan = np.isnan(Y[:, i])
            Y[ind_nan, i] = m[i]
            if np.alltrue(Y[:, i] == 0): # new added
                Y[0, i] = 1 # new added
        # vars = np.var(Y, axis=0, ddof=1)
        # Y = Y[:, vars > 0] # could cause problem!!!
    # mean-variance map
    if exp_fam_type == "normal":
        V = lambda x: sigma_sq
    elif exp_fam_type == "poisson":
        V = lambda x: x
    elif exp_fam_type == "binomial":
        V = lambda x: x * (1 - x / bino_n)

    Y_bar = np.mean(Y, axis=0)
    D_n = V(Y_bar)

    Y_c = Y - Y_bar # center data
    Y_w = n**(-0.5) * Y_c * D_n**(-0.5) # Y_w^T Y_w = S_h + I_p

    U, sval, Vh = np.linalg.svd(Y_w, full_matrices=False)
    V = Vh.conj().T # eigenvectors of Y_w^T Y_w, also eigenvectors of S_h
    white_eval = sval**2 # eigenvalues of Y_w^T Y_w, also (eigenvalues of S_h) + 1
    E = sval[:r]**2
    # white_shr_eval = op_norm_shrink(E, gamma) - 1
    white_shr_eval = op_norm_shrink2(E, gamma) - 1 # shrinked eigenvalues of S_h

    white_V = V[:, :r]
    white_covar = white_V @ np.diag(white_shr_eval) @ white_V.conj().T # S_{h,η}
    S_recolored = np.diag(D_n**0.5) @ white_covar @ np.diag(D_n**0.5) # S_{he}
    white_U = U[:, :r]

    recolor_eval, recolor_V = np.linalg.eigh(S_recolored) # S_h
    recolor_V = np.flip(recolor_V[:, -r:], axis=1)
    recolor_eval = np.flip(recolor_eval[-r:])

    # scaled estimator of the true spike
    c2 = standard_spiked_forward(white_shr_eval, gamma)[1]
    s2 = 1 - c2
    tau = (np.sum(D_n) * white_shr_eval) / (p * recolor_eval)
    alpha = np.ones(r)
    idx = c2 > 0
    alpha[idx] = (1 - s2[idx] * tau[idx]) / c2[idx]
    white_shr_eval_scaled = alpha * recolor_eval
    estim_SNR_improvement = tau / alpha

    return S_recolored, white_shr_eval, white_V, white_U, white_eval, recolor_eval, recolor_V, D_n, white_shr_eval_scaled, estim_SNR_improvement


def wiener_filter(cov_est: np.array, Y: np.array, exp_fam_type="normal", sigma_sq=1, bino_n=1, reg="ridge") -> np.array:
    """Denoise.

    Given the noisy data with its covariance estimated, computes the estimated clean data, using the best linear estimator. This is described in Sec. 5. "Denoiseing" in the ePCA paper. Wiener filter = BLP.

    Parameters
    ----------
    cov_est : ndarray of shape (p, p)
        Estimated covariance matrix.
    Y : ndarray of shape (n, p)
        nxp data matrix with n rows containing p-dimensional noisy signal vectors.
    exp_fam_type : str, optional
        Distribution of the entries of Y, a type of exponential family, ['normal'|'poisson'|'binomial'], by default "normal".
    sigma_sq : int, optional
        Parameters for Gaussian distribution, by default 1.
    bino_n : int, optional
        Parameters for binomial distribution, by default 1.
    reg : str, optional
        Regularization approach, ["ridge"|None].

    Returns
    -------
    X : ndarray of shape (n, p)
        Clean data.
    """

    n, p = Y.shape

    # mean-variance map
    if exp_fam_type == "normal":
        V = lambda x: sigma_sq
    elif exp_fam_type == "poisson":
        V = lambda x: x
    elif exp_fam_type == "binomial":
        V = lambda x: x * (1 - x / bino_n)

    Y_bar = np.mean(Y, axis=0)
    D_n = V(Y_bar)

    if reg == "ridge":
        eps = 0.1
        m = np.mean(D_n)
        Sigma_reg = cov_est + (1-eps) * np.diag(D_n) + eps * m * np.identity(p)
        X = cov_est @ np.linalg.solve(Sigma_reg, Y.T) + (np.diag(D_n) @ np.linalg.solve(Sigma_reg, Y_bar))[:, np.newaxis]
    elif reg is None:
        X = cov_est @ np.linalg.solve(np.diag(D_n) + cov_est, Y.T) + (np.diag(D_n) @ np.linalg.solve(np.diag(D_n) + cov_est, Y_bar))[:, np.newaxis]
    return X.T




'''
Adapted from https://bitbucket.org/mackelab/pop_spike_dyn/src/master/, David Pfau, Eftychios A Pnevmatikakis, and Liam Paninski. Robust learning of low-dimensional dynamics from large neural ensembles. In Adv neur inf proc sys, pages 2391–2399, 2013.
Original idea is from M. Collins, S. Dasgupta, and R. E. Schapire, “A generalization of principal component analysis to the exponential family,” Advances in neural information processing systems, vol. 14, 2001.
'''

def poisson_pca(X: np.array, n_components: int, penalty=0.1):
    """Exponential Poisson PCA.

    Exponential family PCA's Poisson distribution. X ~ Poisson(g(η)), where g is exp in Poisson distribution, η is the natural parameter, and η = X_new V^H, where columns of V is the new orthogonal basis explaining variances in descending order.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features) or say (n_timebins, n_neurons)
        Observed spike counts from Poisson(exp(X_new V^H)).
    n_components : int
        Desired low dimensionality of the latent.
    penalty : float, optional
        Penalty for the norm of X_new and V, by default 0.1.

    Returns
    -------
    X_new : ndarray of shape (n_samples, n_components) or say (n_timebins, n_components)
        Coordinators of each sample projected with basis V.
    V : ndarray of shape (n_features, n_components)
        Orthogonal basis in low dimensional space.
    d : ndarray of shape (n_features,)
        Mean offset.
    """

    rng = np.random.default_rng()
    n_samples, n_features = X.shape

    X_mean = np.mean(X, axis=0)
    __, __, vh_X = np.linalg.svd(X - X_mean)

    X_new_init = 0.01 * rng.normal(size=(n_samples, n_components))
    V_init = vh_X[:, :n_components]
    d_init = np.log(np.maximum(X_mean, 0.1))

    def pack(X_new, V, d):
        return np.hstack((np.reshape(X_new, -1), np.reshape(V, -1), d))

    def unpack(x):
        X_new = np.reshape(x[:n_samples*n_components], (n_samples, n_components))
        V = np.reshape(x[n_samples*n_components:-n_features], (n_features, n_components))
        d = x[-n_features:]
        return X_new, V, d

    def fun(x):
        X_new, V, d = unpack(x)
        theta = X_new @ V.T + d
        g_theta = np.exp(theta)
        temp = g_theta - X # (n_samples, n_features)
        dX_new = temp @ V + penalty * X_new
        dV = temp.T @ X_new + penalty * V
        dd = np.sum(temp, axis=0)
        f = np.sum(-X * theta + np.exp(theta)) + penalty/2*(np.linalg.norm(X_new, ord='fro') + np.linalg.norm(V, ord='fro'))
        g = pack(dX_new, dV, dd)
        return f, g

    x0 = pack(X_new_init, V_init, d_init)
    res = minimize(fun, x0, jac=True, method="L-BFGS-B")
    X_new, V, d = unpack(res.x)
    X_mean = np.mean(X_new, axis=0)
    X_new -= X_mean
    d += V @ X_mean

    return X_new, V, d