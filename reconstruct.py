# GP-based reconstruction of 2D images and 3D spectroscopic data
# Author: Maxim Ziatdinov (email: maxim.ziatdinov@ai4microcopy.com)

# imports
import argparse
import os
import numpy as np
from gprocess import gpr, gprutils
import torch
torch.set_default_tensor_type(torch.DoubleTensor)

# filepath and GP regression parameters
parser = argparse.ArgumentParser("Gaussian processes for sparse 3D data")
parser.add_argument("FILEPATH", nargs="?", type=str,
                    help="provide 3D numpy array of spectroscopic data")
parser.add_argument("--KERNEL", nargs="?", default="Matern52", type=str)
parser.add_argument("--LENGTH_CONSTR_MIN", nargs="?", default=1, type=int)
parser.add_argument("--LENGTH_CONSTR_MAX", nargs="?", default=20, type=int)
parser.add_argument("--LEARNING_RATE", nargs="?", default=0.05, type=float)
parser.add_argument("--INDUCING_POINTS", nargs="?", default=250, type=int)
parser.add_argument("--NORMALIZE", nargs="?", default=1, type=int,
                    help="Normalizes to [0, 1]. 1 is True, 0 is False")
parser.add_argument("--STEPS", nargs="?", default=1000, type=int)
parser.add_argument("--NUM_BATCHES", nargs="?", default=200, type=int)
parser.add_argument("--PROB", nargs="?", default=0.0, type=float,
                    help="Value between 0 and 1." +
                    "Controls number of data points to be removed.")
parser.add_argument("--USE_GPU", nargs="?", default=1, type=int,
                    help="1 for using GPU, 0 for running on CPU")
parser.add_argument("--SAVEDIR", nargs="?", default="Output", type=str,
                    help="directory to save outputs")

args = parser.parse_args()

# Load "ground truth" data (N x M x L spectroscopic grid)
R_true = np.load(args.FILEPATH)
if args.NORMALIZE and np.isnan(R_true).any() == False:
    R_true = (R_true - np.amin(R_true))/np.ptp(R_true)
# Get "ground truth" grid indices
if np.ndim(R_true) == 2:
    e1, e2 = R_true.shape
    c1, c2 = np.mgrid[:e1:1., :e2:1.]
    X_true = np.array([c1, c2])
elif np.ndim(R_true) == 3:
    e1, e2, e3 = R_true.shape
    c1, c2, c3 = np.mgrid[:e1:1., :e2:1., :e3:1.]
    X_true = np.array([c1, c2, c3])
else:
    raise NotImplementedError("The input ndarray must be 2D or 3D")
# Construct lengthscale constraints for all dimensions
LENGTH_CONSTR = [
                 [float(args.LENGTH_CONSTR_MIN) for i in range(np.ndim(R_true))],
                 [float(args.LENGTH_CONSTR_MAX) for i in range(np.ndim(R_true))]
]
# Corrupt data (if args.PROB > 0)
X, R = gprutils.corrupt_data_xy(X_true, R_true, args.PROB)
# Directory to save results
if not os.path.exists(args.MDIR):
    os.makedirs(args.MDIR)
# Reconstruct the corrupt data. Initalize our "reconstructor" first.
reconstr = gpr.reconstructor(
    X, R, X_true, args.KERNEL, LENGTH_CONSTR, args.INDUCING_POINTS,
    ldim=np.ndim(R_true), use_gpu=args.USE_GPU, verbose=True)
# Model training and prediction
mean, sd, hyperparams = reconstr.run(
    args.LEARNING_RATE, args.STEPS, args.NUM_BATCHES)
# Save results
np.savez(os.path.join(args.SAVEDIR, os.path.basename(
    os.path.splitext(args.FILEPATH)[0])+'gpr_reconstruction.npz'),
    original_data=R_true, input_data=R, mean=mean, SD=sd,
    hyperparams=hyperparams)
