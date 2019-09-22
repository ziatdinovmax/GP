import os
import numpy as np
import gpr
import gprutils

# Load data
R_true = np.load()
X_true = np.load()

# directory to store the results
mdir = 'Output'
if not os.path.exists(mdir):
    os.makedirs(mdir)

# number of total steps
exporatory_steps = 45
# edge regions not considered for max uncertainty evaluation
dist_edge = [6, 6]
# learning rate for each iteration (decrease if it becomes unstable)
lr = .1
# size of measurements
msize = 2

# Get initial data. This should be N x M x L spectroscopic grid,
# where only < 5 % of the grid is measured (at random points).
# Here we achieve this by removing 95 % of the "ground truth" data
X, R = gprutils.corrupt_data_xy(X_true, R_true, prob=.95)

# Run exploratory analysis
amax_all, mean_all, sd_all, R_all = [], [], [], []
for i in range(exporatory_steps):
    amax, mean, sd = gpr.exploration_step(X, R, dist_edge, learning_rate=lr)
    # store intermediate results
    amax_all.append(amax)
    mean_all.append(mean)
    sd_all.append(sd)
    R_all.append(R)
    # make a "measurement" in the point with maximum uncertainty
    print('Doing "measurement"...\n')
    R, X = gprutils.do_measurement(R_true, X_true, R, X, amax, msize)
    # (over)write results to disk
    np.save(os.path.join(mdir, 'sgpr_cits_R_5.npy'), R_all)
    np.save(os.path.join(mdir, 'sgpr_cits_means_5.npy'), mean_all)
    np.save(os.path.join(mdir, 'sgpr_cits_sd_5.npy'), sd_all)
    np.save(os.path.join(mdir, 'sgpr_cits_amax_5.npy'), np.array(amax_all))
