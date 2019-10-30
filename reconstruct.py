# GP-based reconstruction of 3D spectroscopic data
# Author: Maxim Ziatdinov (email: maxim.ziatdinov@ai4microcopy.com)

# imports
import argparse
import os
import numpy as np
import gpr
import gprutils
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
parser.add_argument("--STEPS", nargs="?", default=1000, type=int)
parser.add_argument("--PROB", nargs="?", default=0.0, type=float,
                    help="Value between 0 and 1." +
                    "Controls number of data points to be removed.")
parser.add_argument("--USE_GPU", nargs="?", default=True, type=bool)
parser.add_argument("--MDIR", nargs="?", default="Output", type=str,
                    help="directory to save outputs")

args = parser.parse_args()

# Construct lengthscale constraints for all 3 dimensions
LENGTH_CONSTR = [
                 [float(args.LENGTH_CONSTR_MIN) for i in range(3)],
                 [float(args.LENGTH_CONSTR_MAX) for i in range(3)]
]
# Load "ground truth" data (N x M x L spectroscopic grid)
R_true = np.load(args.FILEPATH)
R_true = (R_true - np.amin(R_true))/np.ptp(R_true)
# Get "ground truth" grid indices
e1, e2, e3 = R_true.shape
c1, c2, c3 = np.mgrid[:e1:1., :e2:1., :e3:1.]
X_true = np.array([c1, c2, c3])
# Corrupt data
X, R = gprutils.corrupt_data_xy(X_true, R_true, args.PROB)
# Directory to save results
if not os.path.exists(args.MDIR):
    os.makedirs(args.MDIR)
# Reconstruct the corrupt data. Initalize our "reconstructor" first.
recnstr = gpr.reconstructor(
    X, R, X_true, args.KERNEL, LENGTH_CONSTR, args.INDUCING_POINTS,
    use_gpu=args.USE_GPU, verbose=True)
# Model training
model, losses, hyperparams = recnstr.train_sgpr_model(
    args.LEARNING_RATE, args.STEPS)
# Model prediction
mean, sd = recnstr.sgpr_predict(model, num_batches=200)
# Save results
np.savez(os.path.join(args.MDIR, 'sgpr_reconstruction.npz'),
         mean=mean, sd=sd, hyperparams=hyperparams)
