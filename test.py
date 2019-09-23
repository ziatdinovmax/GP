# Test module
# Author: Maxim Ziatdinov (email: maxim.ziatdinov@ai4microcopy.com)

# imports
import numpy as np
import gpr
import gprutils
import torch
torch.set_default_tensor_type(torch.DoubleTensor)

# number of total steps
STEPS = 45
# edge regions not considered for max uncertainty evaluation
DIST_EDGE = [6, 6]
# learning rate for each iteration (decrease if it becomes unstable)
LR = .1
# size of measurements
MSIZE = 2

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
uncert_idx_all, uncert_val_all = [], []
for i in range(STEPS):
    print('Exploration step {}/{}'.format(i, STEPS))
    # use different bounds on lengthscale at the very beginning
    lscale = None if i < 10 else [[1., 1., 2.5], [4., 4., 10.]]
    # Do exploration step
    uncert_idx, uncert_val = gpr.exploration_step(
        X, R, X_true, DIST_EDGE, lscale, LR)
    uncert_idx, uncert_val = gprutils.checkvalues(
        uncert_idx, uncert_idx_all, uncert_val)
    # store intermediate results
    uncert_idx_all.append(uncert_idx)
    uncert_val_all.append(uncert_val)
    # make a "measurement" in the point with maximum uncertainty
    print('Doing "measurement"...\n')
    R, X = gprutils.do_measurement(R_true, X_true, R, X, uncert_idx, MSIZE)
