# Utility functions
# Author: Maxim Ziatdinov (email: maxim.ziatdinov@ai4microcopy.com)

import os
import numpy as np
import torch
import pyro
import matplotlib.pyplot as plt


def max_uncertainty(sd, dist_edge):
    """
    Finds first 100 points with maximum uncertainty

    Args:
        sd: (N,) ndarray
            predictive SD (N is equal to number of observation points)
        dist_edge: list of two integers
            edge regions not considered for max uncertainty evaluation

    Returns:
        list of indices corresponding to the max uncertainty point
    """
    # sum along the last dimension
    e1, e2 = sd.shape[:2]
    sd = np.sum(sd, axis=-1)
    # mask the edges
    mask = np.zeros(sd.shape, bool)
    mask[dist_edge[0]:e1-dist_edge[0],
         dist_edge[1]:e2-dist_edge[1]] = True
    sd = sd * mask
    # find first 100 points with the largest uncertainty
    amax_list, uncert_list = [], []
    for i in range(100):
        amax = [i[0] for i in np.where(sd == sd.max())]
        amax_list.append(amax)
        uncert_list.append(sd.max())
        sd[amax[0], amax[1]] = 0

    return amax_list, uncert_list


def checkvalues(uncert_idx_list, uncert_idx_all, uncert_val_list):
    """
    Checks if the indices were already used
    (helps not to get stuck in one point)

    Args:
        uncert_idx_list: list of lists with integers
            indices of max uncertainty points for one measurement;
            the list is ordered (max uncertainty -> min uncertainty)
        uncert_idx_all: list of lists with integers
            indices of the already selected points from previous measurements
        uncert_val_list: list with floats
            SD values for each index in uncert_idx_list
            (ordered as max uncertainty -> min uncertainty)

    Returns:
        If no previous occurences found,
        returns the first element in the input list (uncert_idx_list).
        Otherwise, returns the next/closest value from the list.
    """

    _idx = 0
    print('Maximum uncertainty of {} at {}'.format(
        uncert_val_list[_idx], uncert_idx_list[_idx]))
    if len(uncert_idx_all) == 0:
        return uncert_idx_list[_idx], uncert_val_list[_idx]
    while 1 in [1 for a in uncert_idx_all if a == uncert_idx_list[_idx]]:
        print("Finding the next max point...")
        _idx = _idx + 1
        print('Maximum uncertainty of {} at {}'.format(
            uncert_val_list[_idx], uncert_idx_list[_idx]))
    return uncert_idx_list[_idx], uncert_val_list[_idx]


def do_measurement(R_true, X_true, R, X, uncertmax, measure):
    """
    Makes a "measurement" by opening a part of a ground truth
    when working with already acquired or synthetic data

    Args:
        R_true: N x M x L ndarray
            datacube with full observations ('ground truth')
        X_true: N x M x L x c ndarray
            grid indices for full observations
            c is number of dimensions (for xyz coordinates, c = 3)
        R: N x M x L ndarray
            datacube with partial observations (missing values are NaNs)
        X: N x M x L x c ndarray
            grid indices for partial observations (missing points are NaNs)
            c is number of dimensions (for xyz coordinates, c = 3)
        uncertmax: list
            indices of point with maximum uncertainty
            (as determined by GPR model)
        measure: int
            half of measurement square
    """
    print(uncertmax)
    a0, a1 = uncertmax
    # make "observation"
    R_obs = R_true[a0-measure:a0+measure, a1-measure:a1+measure, :]
    X_obs = X_true[:, a0-measure:a0+measure, a1-measure:a1+measure, :]
    # update the input
    R[a0-measure:a0+measure, a1-measure:a1+measure, :] = R_obs
    X[:, a0-measure:a0+measure, a1-measure:a1+measure, :] = X_obs
    return R, X


def prepare_training_data(X, y):
    """
    Reshapes and converts data to torch tensors for GP analysis

    Args:
        X:  c x  N x M x L ndarray
            Grid indices.
            c is equal to the number of coordinate dimensions.
            For example, for xyz coordinates, c = 3.
        y: N x M x L ndarray
            Observations (data points)

    Returns:
        torch tensors with dimensions (M*N*L, c) and (N*M*L,)
    """

    tor = lambda n: torch.from_numpy(n)
    X = X.reshape(X.shape[0], X.shape[1]*X.shape[2]*X.shape[3]).T
    X = tor(X[~np.isnan(X).any(axis=1)])
    y = tor(y.flatten()[~np.isnan(y.flatten())])

    return X, y


def prepare_test_data(X):
    """
    Reshapes and converts data to torch tensors for GP analysis

    Args:
        X:  c x  N x M x L ndarray
            Grid indices.
            c is equal to the number of coordinate dimensions.
            For example, for xyz coordinates, c = 3.

    Returns:
        torch tensor with dimensions (N*M*L, c)
    """

    X = X.reshape(X.shape[0], X.shape[1]*X.shape[2]*X.shape[3]).T

    return X


def corrupt_data_xy(X_true, R_true, prob=0.5):
    """
    Replaces certain % of data with NaNs.
    Applies differently in xy and in z dimensions.
    Specifically, for every corrupted (x, y) point
    we remove all z values associated with this point.

    Args:
        X: c x N x M x L ndarray
           Grid indices.
           c is equal to the number of coordinate dimensions.
           For example, for xyz coordinates, c = 3.
        R_true: N x M x L ndarray
            hyperspectral dataset
        prob: float between 0. and 1.
            controls % of data in xy plane to be corrupted

    Returns:
        c x M x N x L ndarray of grid coordinates
        and M x N x L ndarray of observatons where
        certain % of points is replaced with NaNs
        (note that for every corrupted (x, y) point
        we remove all z values associated with this point)
    """

    pyro.set_rng_seed(0)
    e1, e2, e3 = R_true.shape
    brn = pyro.distributions.Bernoulli(prob)
    indices = [i for i in range(e1*e2) if brn.sample() == 1]
    R = R_true.copy().reshape(e1*e2, e3)
    R[indices, :] = np.nan
    R = R.reshape(e1, e2, e3)
    X = X_true.copy().reshape(3, e1*e2, e3)
    X[:, indices, :] = np.nan
    X = X.reshape(3, e1, e2, e3)
    return X, R

