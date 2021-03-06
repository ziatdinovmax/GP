# GP sample exploration based on maximal uncertainty reducton
# for 3D hyperspectral measurements.
# Currently runs only "synthetic experiments" where
# one needs to provide a full dataset (no missing values) as ground truth.

# Author: Maxim Ziatdinov (email: maxim.ziatdinov@ai4microcopy.com)

# imports
import argparse
import os
import copy
import numpy as np
from gprocess import gpr, gprutils
import torch
torch.set_default_tensor_type(torch.DoubleTensor)

# filepath and GP regression parameters
parser = argparse.ArgumentParser(
    description="Gaussian processes-based sample exploration")
parser.add_argument("FILEPATH", nargs="?", type=str,
                    help="Provide 3D numpy array of spectroscopic data")
parser.add_argument("--ESTEPS", nargs="?", default=65, type=int,
                    help="Number of exploration steps")
parser.add_argument("--MSIZE", nargs="?", default=0, type=int,
                    help="Size of measurements (2*MSIZE+1)")
parser.add_argument("--KERNEL", nargs="?", default="Matern52", type=str)
parser.add_argument("--LENGTH_CONSTR_MIN", nargs="?", default=1, type=int)
parser.add_argument("--LENGTH_CONSTR_MAX", nargs="?", default=20, type=int)
parser.add_argument("--LEARNING_RATE", nargs="?", default=0.1, type=float,
                    help="Learning rate for each exploration step" +
                    "(decrease if it becomes unstable)")
parser.add_argument("--INDUCING_POINTS_RATIO", nargs="?", default=20, type=int,
                    help="ratio of total number of data points" +
                    "to number of inducing points")
parser.add_argument("--NORMALIZE", nargs="?", default=1, type=int,
                    help="Normalizes to [0, 1]. 1 is True, 0 is False")
parser.add_argument("--STEPS", nargs="?", default=500, type=int,
                    help="Number of SVI steps during model training")
parser.add_argument("--USE_GPU", nargs="?", default=1, type=int,
                    help="1 for using GPU, 0 for running on CPU")
parser.add_argument("--SAVEDIR", nargs="?", default="Output", type=str,
                    help="Directory to save outputs")
args = parser.parse_args()

# Load "ground truth" data (N x M x L spectroscopic grid)
# (in real experiment we will just get an empty array)
R_true = np.load(args.FILEPATH)
if args.NORMALIZE and np.isnan(R_true).any() is False:
    R_true = (R_true - np.amin(R_true))/np.ptp(R_true)
# Get "ground truth" grid indices
e1, e2, e3 = R_true.shape
c1, c2, c3 = np.mgrid[:e1:1., :e2:1., :e3:1.]
X_true = np.array([c1, c2, c3])
# Make initial set of measurements for exploration analysis.
# Let's start with "opening" several points along each edge
R = R_true*0
R[R==0] = np.nan
R = gprutils.open_edge_points(R, R_true)
X, R = gprutils.corrupt_data_xy(X_true, R)
dist_edge = [0, 0] # set to non-zero vals when edge points are not "opened"
# Construct lengthscale constraints for all 3 dimensions
LENGTH_CONSTR = [
                 [float(args.LENGTH_CONSTR_MIN) for i in range(3)],
                 [float(args.LENGTH_CONSTR_MAX) for i in range(3)]
]
# Run exploratory analysis
uncert_idx_all, uncert_val_all, mean_all, sd_all, R_all = [], [], [], [], []
if not os.path.exists(args.SAVEDIR): os.makedirs(args.SAVEDIR)
indpts_r = args.INDUCING_POINTS_RATIO
for i in range(args.ESTEPS):
    print('Exploration step {}/{}'.format(i, args.ESTEPS))
    # Make the number of inducing points dependent on the number of datapoints
    indpoints = len(gprutils.prepare_training_data(X, R)[0]) // indpts_r
    # clip to make sure it fits into GPU memory
    indpoints = 2000 if indpoints > 2000 else indpoints
    # Initialize explorer
    bexplorer = gpr.reconstructor(
        X, R, X_true, args.KERNEL, LENGTH_CONSTR,
        indpoints, 3, args.LEARNING_RATE, args.STEPS,
        use_gpu=args.USE_GPU)
    # get indices/value of a max uncertainty point
    uncert_idx, uncert_val, mean, sd = bexplorer.step(dist_edge)
    # some safeguards (to not stuck at one point)
    uncert_idx, uncert_val = gprutils.checkvalues(
        uncert_idx, uncert_idx_all, uncert_val)
    # store intermediate results
    uncert_idx_all.append(uncert_idx)
    uncert_val_all.append(uncert_val)
    R_all.append(copy.deepcopy(R.flatten()))
    mean_all.append(copy.deepcopy(mean))
    sd_all.append(copy.deepcopy(sd))
    # make a "measurement" in the point with maximum uncertainty
    print('Doing "measurement"...\n')
    R, X = gprutils.do_measurement(R_true, X_true, R, X, uncert_idx, args.MSIZE)
    # (over)write results on disk
    np.savez(os.path.join(args.SAVEDIR, os.path.basename(
             os.path.splitext(args.FILEPATH)[0])+'-explorative_analysis.npz'),
             R_all=R_all, mean_all=mean_all,
             sd_all=sd_all, uncert_idx_all=uncert_idx_all)
