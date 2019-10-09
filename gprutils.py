# Utility functions
# Author: Maxim Ziatdinov (email: maxim.ziatdinov@ai4microcopy.com)

import os
import copy
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
    a0, a1 = uncertmax
    # make "observation"
    R_obs = R_true[a0-measure:a0+measure+1, a1-measure:a1+measure+1, :]
    X_obs = X_true[:, a0-measure:a0+measure+1, a1-measure:a1+measure+1, :]
    # update the input
    R[a0-measure:a0+measure+1, a1-measure:a1+measure+1, :] = R_obs
    X[:, a0-measure:a0+measure+1, a1-measure:a1+measure+1, :] = X_obs
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
    X = torch.from_numpy(X)

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


def plot_raw_data(raw_data, slice_number, pos, spec_window=2):
    """
    Plots hyperspectral data as 2D image
    integrated over a certain range of energy/frequency
    and selected individual spectroscopic curves

    Args:
        raw_data: 3D numpy array
            hyperspectral cube
        slice_number: int
            slice from datacube to visualize
        pos: list of lists
            list with [x, y] coordinates of points where
            single spectroscopic curves will be extracted and visualized
        spec_window: int
            window to integrate over in frequency dimension (for 2D "slices")

    """
    # colors sequence
    my_colors = ['gray', 'red', 'green', 'black', 'orange', 'blue']
    # Plotting
    s = slice_number
    spw = spec_window
    _, ax = plt.subplots(1, 2, figsize=(10, 4.5))
    ax[0].imshow(np.sum(raw_data[:, : , s-spw:s+spw], axis=-1), cmap='magma')
    for p, col in zip(pos, my_colors):
        ax[0].scatter(p[1], p[0], c=col)
        ax[1].plot(raw_data[p[0], p[1], :], c=col)
    ax[1].axvspan(s-spw, s+spw linestyle = '--')
    ax[0].set_title('Grid spectroscopy')
    ax[1].set_title('Individual spectroscopic curve')
    plt.subplots_adjust(wspace=.3)
    plt.show()


def plot_reconstructed_data(R, mean, sd, R_true,
                            slice_number, pos,
                            spec_window=2, **kwargs):
    """
    R: 3D numpy array
        hyperspectral cube (input data for GP regression)
    mean: 1D numpy array
        predictive mean
        (flattened; actual dimensions are the same as for R and R_true)
    sd: 1D numpy array
        standard deviation
        (flattened; actual dimensions are the same as for R and R_true)
    R_true: 3D numpy array
        hyperspectral cube
        (original data; if no observations weere removed, R_true == R)
    slice_number: int
        slice from datacube to visualize
    pos: list of lists
        list with [x, y] coordinates of points where
        single spectroscopic curves will be extracted and visualized
    spec_window: int
        window to integrate over in frequency dimension (for 2D "slices")

    **Kwargs:
        sparsity: float (between 0 and 1)
            indicates % of data points removed (used only for figure title)
    """
    sparsity = kwargs.get('sparsity')
    s = slice_number
    e1, e2, e3 = R_true.shape
    spw = spec_window
    Rtest = mean.reshape(e1, e2, e3)
    R_sd = sd.reshape(e1, e2, e3)
    my_colors = ['black', 'red', 'green', 'gray', 'orange', 'blue']
    fig, ax = plt.subplots(3, 2, figsize=(10, 16))
    ax[0, 0].imshow(
        np.sum(R_true[:, :, s-spw:s+spw], axis=-1), cmap='nipy_spectral')
    for p, col in zip(pos, my_colors):
        ax[0, 0].scatter(p[1], p[0], c=col)
        ax[0, 1].plot(R_true[p[0], p[1], :], c=col)
    ax[0, 1].axvspan(s-spw, s+spw, linestyle='--', alpha=.1)
    ax[0, 1].set_ylim(-0.1, 0.9)
    for _ax in [ax[0, 0], ax[0, 1]]:
        _ax.set_title('Ground truth')
    ax[1, 0].imshow(
        np.sum(R[:, :, s-spw:s+spw], axis=-1), cmap='nipy_spectral')
    for p, col in zip(pos, my_colors):
        ax[1, 0].scatter(p[1], p[0], c=col)
        ax[1, 1].plot(R[p[0], p[1], :], c=col)
    ax[1, 1].axvspan(s-spw, s+spw, linestyle='--', alpha=.1)
    ax[1, 1].set_ylim(-0.1, 0.9)
    for _ax in [ax[1, 0], ax[1, 1]]:
        if sparsity:
            _ax.set_title(
                'Corrupted data\n{}% of observations removed'.format(sparsity*100))
        else:
            _ax.set_title('Corrupted data')
    ax[2, 0].imshow(
        np.sum(Rtest[:, :, s-spw:s+spw], axis=-1), cmap='nipy_spectral')
    for p, col in zip(pos, my_colors):
        ax[2, 0].scatter(p[1], p[0], c=col)
        ax[2, 1].plot(Rtest[p[0], p[1], :], c=col)
        ax[2, 1].fill_between(np.arange(e3),
                        (Rtest[p[0], p[1], :] -
                         2.0 * R_sd[p[0], p[1], :]),
                        (Rtest[p[0], p[1], :]
                         + 2.0 * R_sd[p[0], p[1], :]),
                        color=col, alpha=0.15)
    ax[2, 1].axvdpsn(s-spw, s+spw, linestyle='--', alpha=.1)
    ax[2, 1].set_ylim(-0.1, 0.9)
    for _ax in [ax[2, 0], ax[2, 1]]:
        _ax.set_title('GPR reconstruction')
    plt.subplots_adjust(hspace=.3)
    plt.show()


def plot_exploration_results(R_all, mean_all, sd_all, R_true,
                             episodes, slice_number, pos, dist_edge,
                             spec_window=2, mask_predictions=False):
    """
    Plots predictions at different stages ("episodes")
    of max uncertainty-based sample exploration

    Args:
        R_all: list with ndarrays
            Observed data points at each exploration step
        mean_all: list of ndarrays
            Predictive mean at each exploration step
        sd_all:
            Integrated (along energy dimension) SD at each exploration step
        R_true:
            Ground truth data (full observations) for synthetic data
            OR array of zeros/NaNs with N x M x L dims for real experiment
        episodes: list of integers
            list with # of iteration steps to be visualized
        slice_number: int
            slice from datacube to visualize
        pos: list of lists
            list with [x, y] coordinates of points where
            single spectroscopic curves will be extracted and visualized
        dist_edge: list with two integers
            this should be the same as in exploration analysis
        spec_win: int
            window to integrate over in frequency dimension (for 2D "slices")
        mask_predictions: bool
            mask edge regions not used in max uncertainty evaluation
            in predictive mean plots

        Returns:
            Plot the results of exploration analysis for the selected steps
    """

    s = slice_number
    spw = spec_window
    _colors = ['black', 'red', 'green', 'blue', 'orange']
    e1, e2, e3 = R_true.shape

    # plot ground truth data if available
    if not np.isnan(R_true).any() or np.unique(R_true).any():
        _, ax = plt.subplots(1, 2, figsize=(7, 3))
        ax[0].imshow(np.sum(R_true[:, :, s-spw:s+spw], axis=-1), cmap='jet')
        for p, col in zip(pos, _colors):
            ax[0].scatter(p[1], p[0], c=col)
            ax[1].plot(R_true[p[0], p[1], :], c=col)
        ax[1].axvspan(s-spw, s+spw, linestyle='--', alpha=.15)
        ax[0].set_title('Grid spectroscopy\n(ground truth)')
        ax[1].set_title('Individual spectroscopic curves\n(ground truth)')

    # Plot predictions
    n = len(episodes) + 1
    fig = plt.figure(figsize=(20, 16))

    for i in range(1, n):
        Rcurr = R_all[episodes[i-1]].reshape(e1, e2, e3)
        Rtest = mean_all[episodes[i-1]].reshape(e1, e2, e3)
        R_sd = sd_all[episodes[i-1]].reshape(e1, e2, e3)

        ax = fig.add_subplot(4, n, i)
        ax.imshow(np.sum(Rcurr[:, :, s-spw:s+spw], axis=-1), cmap='jet')
        ax.set_title('Observations (episode {})'.format(episodes[i-1]))

        ax = fig.add_subplot(4, n, i + n)
        Rtest_to_plot = copy.deepcopy((np.sum(Rtest[:, :, s-spw:s+spw], axis=-1)))
        mask = np.zeros(Rtest_to_plot.shape, bool)
        mask[dist_edge[0]:e1-dist_edge[0],
             dist_edge[1]:e2-dist_edge[1]] = True
        if mask_predictions:
            Rtest_to_plot[~mask] = np.nan
        ax.imshow(Rtest_to_plot, cmap='jet')
        for p, col in zip(pos, _colors):
            ax.scatter(p[1], p[0], c=col)
        ax.set_title('GPR reconstruction (episode {})'.format(episodes[i-1]))
        ax = fig.add_subplot(4, n, i + 2*n)
        for p, col in zip(pos, _colors):
            ax.plot(Rtest[p[0], p[1], :], c=col)
            ax.fill_between(np.arange(e3),
                            (Rtest[p[0], p[1], :] - 2.0 *
                            R_sd[p[0], p[1], :]),
                            (Rtest[p[0], p[1], :] + 2.0 *
                            R_sd[p[0], p[1], :]),
                            color=col, alpha=0.15)
            ax.axvspan(s-spw, s+spw, linestyle='--', alpha=.1)
        ax.set_title('GPR reconstruction (episode {})'.format(episodes[i-1]))

        ax = fig.add_subplot(4, n, i + 3*n)
        R_sd_to_plot = copy.deepcopy(R_sd)
        R_sd_to_plot = np.sum(R_sd_to_plot, axis=-1)
        R_sd_to_plot[~mask] = np.nan
        ax.imshow(R_sd_to_plot, cmap='jet')
        ax.set_title('Integrated uncertainty (episode {})'.format(episodes[i-1]))

    plt.subplots_adjust(hspace=.3)
    plt.subplots_adjust(wspace=.3)
    plt.show()
