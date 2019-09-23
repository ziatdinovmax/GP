# Test module
# Author: Maxim Ziatdinov (email: maxim.ziatdinov@ai4microcopy.com)

# imports
import os
import copy
import numpy as np
import gpr
import gprutils
import torch
torch.set_default_tensor_type(torch.DoubleTensor)

# Number of exploratory steps
ESTEPS = 40
# Number of steps for a single model training
STEPS = 1000
# Type of kernel
KERNEL = 'RationalQuadratic'
# Bounds on priors
LENGTH_CONSTR = [[1., 1., 1.], [40., 40., 40.]]
# Edge regions not considered for max uncertainty evaluation
DIST_EDGE = [6, 6]
# Learning rate for each iteration (decrease if it becomes unstable)
LR = .05
# Size of measurements
MSIZE = 2
# Run on CPU or GPU
USEGPU = True
# Directory to save data
MDIR = 'Output'

# Load "ground truth" data (N x M x L spectroscopic grid)
# (in real experiment we will just get an empty array)
R_true = np.load('test_data/nanoscale12_grid_50_50_lockin-subset.npy')
R_true = (R_true - np.amin(R_true))/np.ptp(R_true)
# Get "ground truth" grid indices
e1, e2, e3 = R_true.shape
c1, c2, c3 = np.mgrid[:e1:1., :e2:1., :e3:1.]
X_true = np.array([c1, c2, c3])

# Make initial set of measurements for exploration analysis.
# In real experiment we measure ~5 % of grid at random points
# Here we achieve this by removing 95 % of the "ground truth" data
X, R = gprutils.corrupt_data_xy(X_true, R_true, prob=.95)

# Run exploratory analysis
uncert_idx_all, uncert_val_all, mean_all, sd_all, R_all = [], [], [], [], []
if not os.path.exists(MDIR): os.makedirs(MDIR)
for i in range(ESTEPS):
    print('Exploration step {}/{}'.format(i, ESTEPS))
    # Do exploration step. 'uncert_idx' are the indices of a region with maximum uncertainty
    bexplorer = gpr.explorer(X, R, X_true, KERNEL, LENGTH_CONSTR, use_gpu=USEGPU)
    uncert_idx, uncert_val, mean, sd = bexplorer.step(LR, STEPS, DIST_EDGE)
    # some safeguards (to not stuck at one point)
    uncert_idx, uncert_val = gprutils.checkvalues(
        uncert_idx, uncert_idx_all, uncert_val)
    # store intermediate results 
    uncert_idx_all.append(uncert_idx)
    uncert_val_all.append(uncert_val)
    # (optional)
    R_all.append(copy.deepcopy(R.flatten()))
    mean_all.append(copy.deepcopy(mean))
    sd_all.append(copy.deepcopy(sd))
    # make a "measurement" in the point with maximum uncertainty
    print('Doing "measurement"...\n')
    R, X = gprutils.do_measurement(R_true, X_true, R, X, uncert_idx, MSIZE)
    # (over)write results on disk
    # (optional)
    np.save(os.path.join(MDIR, 'sgpr_cits_R_5.npy'), R_all)
    np.save(os.path.join(MDIR, 'sgpr_cits_means_5.npy'), mean_all)
    np.save(os.path.join(MDIR, 'sgpr_cits_sd_5.npy'), sd_all)
    np.save(os.path.join(MDIR, 'sgpr_cits_amax_5.npy'), uncert_idx_all)
