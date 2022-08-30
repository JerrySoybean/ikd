#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from ikd.utils import cov2dist2, kernel_cov_generator, rigid_transform, align
from sklearn.metrics import r2_score
from scipy.spatial.distance import cdist
from scipy.linalg import eigh


def ikd(cov_samp_th: np.array, d_latent: int, kernel="squared exponential", variance=1, length_scale=1, extra_kernel_hyperparam=None) -> tuple:
    """Inverse Kernel Decomposition.

    Identify latents from filtered sample covariance matrix.

    Parameters
    ----------
    cov_samp_th : ndarray of shape (n_points, n_points)
        Filtered sample covariance matrix.
    d_latent : int
        Latent dimensionality.
    kernel : str, optional
        ["squared exponential" | "rational quadratic" | "gamma-exponential" | "matern"], by default "squared exponential".
    variance : int, optional
        Marginal variance, by default 1.
    length_scale : int, optional
        Length scale of the kernel, by default 1.
    extra_kernel_hyperparam : int, optional
        α in rational quadratic kernel; γ in gamma-exponential kernel; ν in Matérn kernel, by default None.

    Returns
    -------
    z_pred : ndarray of shape (n_points, d_latent)
        Estimated latents.
    contribution : float
        Cumulative contribution of the eigen-decomposition.
    """

    n_points = cov_samp_th.shape[0]
    pairwise_dist2 = cov2dist2(cov_samp_th, kernel=kernel, variance=variance, length_scale=length_scale, extra_kernel_hyperparam=extra_kernel_hyperparam)
    ref_point = np.argmin(pairwise_dist2.max(axis=0))
    # print(ref_point)
    k = (pairwise_dist2[:, [ref_point]] + pairwise_dist2[[ref_point], :] - pairwise_dist2) / 2
    # eigenvalues, eigenvectors = np.linalg.eigh(k)
    # z_pred = eigenvectors[:, -d_latent:] * np.maximum(eigenvalues[-d_latent:], 0)**0.5
    eigenvalues, eigenvectors = eigh(k, subset_by_index=[n_points-d_latent, n_points-1])
    z_pred = eigenvectors * np.maximum(eigenvalues, 0)**0.5
    # z_pred[ref_point], z_pred[:, ref_point] = 0, 0 # for security
    return z_pred


def maximal_cliques(cov_scaled: np.array(float), clique_th: float) -> list:
    """Find maximal cliques using the Bron-Kerbosch algorithm.

    Given a graph's boolean adjacency matrix, A, find all maximal cliques on A using the Bron-Kerbosch algorithm in a recursive manner. The graph is required to be undirected and must contain no self-edges.

    This script is adapted from Jeffery Wildman's script (http://www.mathworks.com/matlabcentral/fileexchange/30413-bron-kerbosch-maximal-clique-finding-algorithm), implementing Adrian Wilder's suggested speed-up (a factor of > 200 speed up on my test data)

    For speed (e.g. I am using this as a subroutine on >1M small subgraphs), there is no input checking, use standard adjacency matrices (undirected, no loops) **
    
    Ref: Bron, Coen and Kerbosch, Joep, "Algorithm 457: finding all cliques of an undirected graph", Communications of the ACM, vol. 16, no. 9, pp: 575?577, September 1973.

    Ref: Cazals, F. and Karande, C., "A note on the problem of reporting maximal cliques", Theoretical Computer Science (Elsevier), vol. 407, no. 1-3, pp: 564-568, November 2008.

    IB: last updated, 3/23/14

    Parameters
    ----------
    cov_scaled : ndarray of shape (n_points, n_points)
        Covariance matrix scaled to have all elements in (0, 1) with all elements in the diagonal to be 1.
    clique_th : float
        Threshold for maximal cliques, in range (0, 1), typically between 0.1 and 0.3.

    Returns
    -------
    clique_list : list[np.array]
        A list in which each element is an indices array as a clique.
    """

    n_points = cov_scaled.shape[0] # number of latents
    r_idx, c_idx = np.where(cov_scaled > clique_th)
    adj_mat = np.zeros_like(cov_scaled, dtype=bool)
    adj_mat[r_idx, c_idx] = True
    adj_mat[range(adj_mat.shape[0]), range(adj_mat.shape[0])] = False
    clique_list = [] # storage for maximal cliques
    r = np.zeros(n_points, dtype=bool) # currently growing clique
    p = np.ones(n_points, dtype=bool) # prospective nodes connected to all nodes in r
    x = np.zeros(n_points, dtype=bool) # nodes already processed

    def bron_kerbosch(r, p, x):
        # BKv2
        if not np.any(np.bitwise_or(p, x)):
            # report r as a maximal clique
            clique_list.append(np.where(r)[0])
        else:
            if len(clique_list) >= 500:
                # print("Terminate the maximal_cliques algorithm")
                return
            # choose pivot
            p_pivots = np.bitwise_or(p, x)
            bin_p = np.zeros(n_points)
            bin_p[p] = 1 # bin_p contains one at indices equal to the values in p
            p_counts = bin_p @ adj_mat[:, p_pivots] # cardinalities of the sets of neighbors of each p_pivots intersected with p
            # select one of the p_pivots with the largest count
            u_p = np.where(p_pivots)[0][np.argmax(p_counts)]

            for u in np.where(np.bitwise_and(np.bitwise_not(adj_mat[u_p]), p))[0]:
                p[u] = False
                r_new = r.copy()
                r_new[u] = True
                n_u = adj_mat[u]
                p_new = np.bitwise_and(p, n_u)
                x_new = np.bitwise_and(x, n_u)
                bron_kerbosch(r_new, p_new, x_new)
                x[u] = True

    bron_kerbosch(r, p, x)
    return clique_list


def combine_two_eig(clique0: np.array, clique1: np.array, eig0: np.array, eig1: np.array, cov_samp_th: np.array, ikd_clique, clique_list_backup, eig_list_backup, record=None, merge_method="to left"):
    """Combine two cliques.

    Parameters
    ----------
    clique0 : 1darray of shape (n_points0,)
        Indices array of the clique0.
    clique1 : 1darray of shape (n_points1,)
        Indices array of the clique1.
    eig0 : ndarray of shape (n_points0, d_latent)
        Estimated latents of clique0.
    eig1 : ndarray of shape (n_points1, d_latent)
        Estimated latents of clique1.
    cov_samp_th : ndarray of shape (n_points, n_points)
        Filtered sample covariance matrix.
    ikd_clique : callable
        IKD that can be applied on clique.
    record : 1darray of shape (n_points,), optional
        Recorded merging history used for weighted merge_method, by default None.
    merge_method : str, optional
        Dealing method about the shared points between two cliques, by default "to left".

    Returns
    -------
    clique : 1darray of shape (n_points2,)
        Indices array of the merged clique0 and clique 1. Union of clique0 and clique1.
    eig : ndarray of shape (n_points2, d_latent)
        Merged latent estimation of clique0 and clique 1.
    """
    
    # e.g., clique0 = [0, 1, 2], clique1 = [1, 2, 3]
    if len(clique1) > len(clique0):
        clique0, clique1 = clique1, clique0
        eig0, eig1 = eig1, eig0
    n_points = cov_samp_th.shape[0]
    inter = np.intersect1d(clique0, clique1) # [1, 2]
    clique = np.union1d(clique0, clique1) # [0, 1, 2, 3]

    if eig0 is None:
        clique_list_backup.append(clique0)
        eig0 = ikd_clique(clique0)
        eig_list_backup.append(eig0)
    if len(clique0) == len(clique):
        return clique0, eig0
    if eig1 is None:
        clique_list_backup.append(clique1)
        eig1 = ikd_clique(clique1)
        eig_list_backup.append(eig1)

    d_latent = eig0.shape[1]
    eig0_onehot = np.zeros((n_points, d_latent))
    eig0_onehot[clique0] = eig0
    eig1_onehot = np.zeros((n_points, d_latent))
    eig1_onehot[clique1] = eig1
    eig = np.zeros((n_points, d_latent))

    if len(inter) >= d_latent+1:
        rh, t = rigid_transform(eig1_onehot[inter], eig0_onehot[inter])
        eig1_onehot[clique1] = eig1_onehot[clique1] @ rh + t

        eig[clique1] = eig1_onehot[clique1]
        eig[clique0] = eig0_onehot[clique0]
        if merge_method == "weighted":
            record[clique1] += 1
            record_inter = record[inter, np.newaxis]
            eig[inter] = eig0_onehot[inter] * (record_inter - 1) / record_inter + eig1_onehot[inter] / record_inter
        elif merge_method == "average":
            eig[inter] = (eig0_onehot[inter] + eig1_onehot[inter]) / 2
        elif merge_method == "to left":
            pass
        else:
            pass
    elif len(inter) == 0:
        print("Two cliques share no points.")
        # print(clique0, clique1)
        cov_samp_th_01 = cov_samp_th[clique0][:, clique1]
        i, j = np.unravel_index(np.argmax(cov_samp_th_01), cov_samp_th_01.shape)
        i, j = clique0[i], clique1[j]
        pos0 = d_latent+2
        pos1 = d_latent+2
        clique0_sub = np.sort(clique0[np.argpartition(cov_samp_th[i, clique0], -pos0)[-pos0:]])
        clique1_sub = np.sort(clique1[np.argpartition(cov_samp_th[j, clique1], -pos1)[-pos1:]])
        clique01 = np.sort(np.hstack((clique0_sub, clique1_sub)))
        eig01 = np.ones((n_points, d_latent))
        eig01[clique01] = ikd_clique(clique01)

        rh, t = rigid_transform(eig1_onehot[clique1_sub], eig01[clique1_sub])
        eig[clique1] = eig1_onehot[clique1] @ rh + t
        rh, t = rigid_transform(eig0_onehot[clique0_sub], eig01[clique0_sub])
        eig[clique0] = eig0_onehot[clique0] @ rh + t
    else:
        print(f"Two cliques share less than (d+1) = {d_latent+1} points.")
        # print(clique0, clique1, inter)
        error = np.zeros(len(inter))
        for i, idx in enumerate(inter):
            clique0_sub = np.sort(clique0[np.argpartition(cov_samp_th[idx, clique0], -(d_latent+2))[-(d_latent+2):]])
            clique1_sub = np.sort(clique1[np.argpartition(cov_samp_th[idx, clique1], -(d_latent+2))[-(d_latent+2):]])
            clique01 = np.unique(np.hstack((clique0_sub, clique1_sub)))
            eig01 = np.ones((n_points, d_latent))
            eig01[clique01] = ikd_clique(clique01)
            rh, t = rigid_transform(eig0_onehot[clique0_sub], eig01[clique0_sub])
            res0 = np.sum((eig0_onehot[clique0_sub] @ rh + t - eig01[clique0_sub])**2, axis=0)
            rh, t = rigid_transform(eig1_onehot[clique1_sub], eig01[clique1_sub])
            res1 = np.sum((eig1_onehot[clique1_sub] @ rh + t - eig01[clique1_sub])**2, axis=0)
            error[i] = np.sqrt(np.sum(res0)) + np.sqrt(np.sum(res1))
        idx = inter[np.argmin(error)]
        clique0_sub = clique0[np.argpartition(cov_samp_th[idx, clique0], -(d_latent+2))[-(d_latent+2):]]
        clique1_sub = clique1[np.argpartition(cov_samp_th[idx, clique1], -(d_latent+2))[-(d_latent+2):]]
        clique01 = np.unique(np.hstack((clique0_sub, clique1_sub)))
        eig01 = np.ones((n_points, d_latent))
        eig01[clique01] = ikd_clique(clique01)

        rh, t = rigid_transform(eig1_onehot[clique1_sub], eig01[clique1_sub])
        eig1_onehot[clique1] = eig1_onehot[clique1] @ rh + t
        rh, t = rigid_transform(eig0_onehot[clique0_sub], eig01[clique0_sub])
        eig0_onehot[clique0] = eig0_onehot[clique0] @ rh + t

        eig[clique1] = eig1_onehot[clique1]
        eig[clique0] = eig0_onehot[clique0]
        eig[inter] = (eig0_onehot[inter] + eig1_onehot[inter]) / 2

    eig = eig[clique]
    return clique, eig


def merge_cliques(clique_list: list, cov_samp_th: np.array, ikd_clique, merge_method="to left"):
    """Merge all cliques.

    Parameters
    ----------
    clique_list : list of 1darrays
        List of all maximal cliques learned by the Bron-Kerbosch algorithm using the threshold sample covariance matrix.
    cov_samp_th : ndarray of shape (n_points, n_points)
        Filtered sample covariance matrix.
    ikd_clique : callable
        IKD that can be applied on clique.
    merge_method : str, optional
        Dealing method about the shared points between two cliques, by default "to left".

    Returns
    -------
    eig : ndarray of (n_points, d_latent)
        All merged latent estimation as a whole.
    """

    eig_list = [None for __ in range(len(clique_list))]
    clique_list_backup = []
    eig_list_backup = []

    # method 1: At each step, select two with max interset to combine
    n_points = cov_samp_th.shape[0]
    clique_onehot_mat = np.zeros((len(clique_list), n_points))
    for i in range(len(clique_list)):
        clique_onehot_mat[i, clique_list[i]] = 1
    inter_mat = clique_onehot_mat @ clique_onehot_mat.T
    inter_mat[range(len(inter_mat)), range(len(inter_mat))] = -1

    while len(clique_list) > 1:
        i, j = np.unravel_index(np.argmax(inter_mat), inter_mat.shape)
        clique_i = clique_list[i]
        clique_j = clique_list.pop(j)
        eig_i = eig_list[i]
        eig_j = eig_list.pop(j)
        clique, eig = combine_two_eig(clique_i, clique_j, eig_i, eig_j, cov_samp_th, ikd_clique, clique_list_backup, eig_list_backup, merge_method=merge_method)
        if len(clique) == n_points: # early stop
            break
        clique_onehot_mat = np.delete(clique_onehot_mat, j, axis=0)
        clique_onehot_mat[i, clique] = 1
        inter_mat = np.delete(np.delete(inter_mat, j, axis=0), j, axis=1)
        inter_mat[i] = clique_onehot_mat[i] @ clique_onehot_mat.T
        inter_mat[:, i] = inter_mat[i]
        inter_mat[i, i] = -1
        clique_list[i] = clique
        eig_list[i] = eig
    
    # # method 2: Continuously select a new clique to combine to the current main clique
    # length_list = [len(clique) for clique in clique_list]
    # record = np.zeros(n_points)
    # start_idx = np.argmax(length_list)
    # clique = clique_list.pop(start_idx)
    # record[clique] = 1
    # eig = eig_list.pop(start_idx)

    # d_observation = len(clique_list)

    # for i in range(d_observation):
    #     inter_count_list = np.zeros(len(clique_list))
    #     for j in range(len(clique_list)):
    #         inter_count_list[j] = len(np.intersect1d(clique, clique_list[j]))
    #     combine_with = np.argmax(inter_count_list)
    #     clique, eig = combine_two_eig(clique, clique_list.pop(combine_with), eig, eig_list.pop(combine_with), n, ikd_clique, record)
    # # print(record)
    return eig, clique_list_backup, eig_list_backup


def estimate_length_scale(pairwise_dist_from_samp, z_pred, cov_samp_th, variance):
    """Re-estimate the length scale in blockwise ikd.

    Parameters
    ----------
    pairwise_dist_from_samp : ndarray of shape (n_points, n_points)
        Pairwise distance estimated from sample covariance matrix.
    z_pred : ndarray of shape (n_points, d_latent)
        Estimated latents.
    cov_samp_th : ndarray of shape (n_points, n_points)
        Filtered sample covariance matrix.
    variance : int, optional
        Marginal variance, by default 1

    Returns
    -------
    length_scale : float
        A value that makes the rebuilt covariance much close to the sample covariance matrix
    """
    pairwise_dist = cdist(z_pred, z_pred) # pairwise distance from cov_samp_th from z_pred
    idx = np.where(np.bitwise_and(cov_samp_th > 0.15 * variance, cov_samp_th < 0.85 * variance))
    length_scale = np.sum(pairwise_dist[idx[0], idx[1]] * pairwise_dist_from_samp[idx[0], idx[1]]) / np.sum(pairwise_dist_from_samp[idx[0], idx[1]]**2)
    return length_scale


def ikd_blockwise(cov_samp_th: np.array, d_latent: int, kernel="squared exponential", variance=1, length_scale=1, extra_kernel_hyperparam=None, clique_th_or_d_observation=0.2, z_ref=None):
    """Blockwise Inverse Kernel Decomposition.

    Parameters
    ----------
    cov_samp_th : ndarray of shape (n_points, n_points)
        Filtered sample covariance matrix.
    d_latent : int
        Latent dimensionality.
    kernel : str, optional
        ["squared exponential" | "rational quadratic" | "gamma-exponential" | "matern"], by default "squared exponential".
    variance : int, optional
        Marginal variance, by default 1.
    length_scale : int, optional
        Length scale of the kernel, by default 1.
    extra_kernel_hyperparam : int, optional
        α in rational quadratic kernel; γ in gamma-exponential kernel; ν in Matérn kernel, by default None.
    clique_th_or_d_observation : float, optional
        If it is >= 1, then it is the GP samples, i.e., the observation dimensionality. Otherwise, it is assumed to be the clique threshold.
    z_ref : ndarray of shape (n_points, d_latent), optional
        A reference latent estimations. If provided, when the rebuilt sample covariance matrix by the blockwise IKD is bad than that, then the merging is unstable, so we align each clique to the corresponding points in z_ref. by default None.

    Returns
    -------
    z_block : ndarray of shape (n_points, d_latent)
        Estimated latents.
    """

    n_points = cov_samp_th.shape[0]
    def ikd_clique(clique):
        return ikd(cov_samp_th[clique][:, clique], d_latent, kernel=kernel, variance=variance, length_scale=length_scale, extra_kernel_hyperparam=extra_kernel_hyperparam)

    # Step 1: Determine clique threshold
    if clique_th_or_d_observation >= 1:
        d_observation = clique_th_or_d_observation
        if d_observation <= 100:
            clique_th = 0.3
        elif d_observation >= 1000:
            clique_th = 0.1
        else:
            clique_th = 0.7 - 0.2 * np.log10(d_observation)
    else:
        clique_th = clique_th_or_d_observation

    # Step 2: Find clique_list
    while True:
        clique_list = maximal_cliques(cov_samp_th / variance, clique_th)
        clique_list = [clique for clique in clique_list if len(clique) >= d_latent + 2]
        if len(clique_list) <= 1:
            print("Only one clique, identical to full eigen-decomposition")
            z_block = ikd_clique(clique_list[0]) # z_ikd
            return z_block
        remaining_indices = np.setdiff1d(np.arange(n_points), np.unique(np.concatenate(clique_list)))
        merge_method = "to left"
        if len(remaining_indices) > 0:
            pos = d_latent + max(int(0.02 * n_points), 2)
            if len(remaining_indices) / n_points <= 0.05:
                print(f"Use nearest neighbors to find cliques that includes those remaining indices: {remaining_indices}")
                clique_list.extend([np.sort(np.argpartition(cov_samp_th[idx], -pos)[-pos:]) for idx in remaining_indices])
                break
            else:
                if clique_th_or_d_observation < 1 or clique_th > 0.26:
                    merge_method = "average"
                    print("Too many remaining indices, use nearest neighbors to find all cliques for every points")
                    clique_list = [np.sort(np.argpartition(cov_samp_th[idx], -pos)[-pos:]) for idx in range(n_points)]
                    break
                else:
                    clique_th += 0.04
                    continue
        else:
            break
    print(f"Clique threshold: {clique_th}, number of cliques: {len(clique_list)}") # debug use

    # Step 3: Do eigen-decomposition blockwisely, and merge them
    z_block, clique_list, eig_list = merge_cliques(clique_list, cov_samp_th, ikd_clique, merge_method)

    # Step 4: Optional step. If z_ref is provided, then the algorithm will compare the recovery with z_ref, and determine if it is necessary to align with z_ref in case z_block from above is really bad.
    if z_ref is not None:
        # estimate length_scale
        pairwise_dist_from_samp = np.sqrt(cov2dist2(cov_samp_th, kernel=kernel, variance=variance, length_scale=1, extra_kernel_hyperparam=extra_kernel_hyperparam))

        length_scale_ref = estimate_length_scale(pairwise_dist_from_samp, z_ref, cov_samp_th, variance=variance)
        cov_ref = kernel_cov_generator(z_ref, kernel=kernel, variance=variance, length_scale=length_scale_ref, extra_kernel_hyperparam=extra_kernel_hyperparam)
        length_scale_block = estimate_length_scale(pairwise_dist_from_samp, z_block, cov_samp_th, variance=variance)
        cov_block = kernel_cov_generator(z_block, kernel=kernel, variance=variance, length_scale=length_scale_block, extra_kernel_hyperparam=extra_kernel_hyperparam)

        if r2_score(cov_samp_th.flatten(), cov_ref.flatten()) - r2_score(cov_samp_th.flatten(), cov_block.flatten()) >= 0.05:
            print("Merge with the reference estimation!")
            z_block = np.zeros((len(clique_list), z_ref.shape[0], z_ref.shape[1]))
            z_block[:, :, :].fill(np.nan)
            for i, (clique_i, eig_i) in enumerate(zip(clique_list, eig_list)):
                z_block[i, clique_i] = align(z_ref[clique_i], eig_i)
            z_block = np.nanmean(z_block, axis=0)

    return z_block