# imports
import os
import numpy as np
import gpr
import gprutils
import torch
torch.set_default_tensor_type(torch.DoubleTensor)

# GP regression parameters
KERNEL = "Matern52"
LENGTH_CONSTR_MIN = 1
LENGTH_CONSTR_MAX = 20
LEARNING_RATE = 0.05
INDUCING_POINTS = 100
STEPS = 100
USE_GPU = True
MDIR = 'Output'

# Construct lengthscale constraints for all 3 dimensions
LENGTH_CONSTR = [
                 [float(LENGTH_CONSTR_MIN) for i in range(3)],
                 [float(LENGTH_CONSTR_MAX) for i in range(3)]
]
# Load "ground truth" data (N x M x L spectroscopic grid)
R_true = np.load('test_data/nanoscale12_grid_50_50_lockin-subset.npy')
R_true = (R_true - np.amin(R_true))/np.ptp(R_true)
# Get "ground truth" grid indices
e1, e2, e3 = R_true.shape
c1, c2, c3 = np.mgrid[:e1:1., :e2:1., :e3:1.]
X_true = np.array([c1, c2, c3])
# Corrupt data
X, R = gprutils.corrupt_data_xy(X_true, R_true, prob=.7)
# Directory to save results
if not os.path.exists(MDIR):
    os.makedirs(MDIR)
# Reconstruct the corrupt data. Initalize our "reconstructor" first.
reconstructor = gpr.explorer(
    X, R, X_true, KERNEL, LENGTH_CONSTR, INDUCING_POINTS,
    use_gpu=USE_GPU, verbose=True)
# Model training
model, losses, hyperparams = reconstructor.train_sgpr_model(
    LEARNING_RATE, STEPS)
# Model prediction
mean, sd = reconstructor.sgpr_predict(model, num_batches=200)
# Save results
np.save(os.path.join(MDIR, 'spgr_reconstr_mean.npy'), mean)
np.save(os.path.join(MDIR, 'spgr_reconstr_sd.npy'), sd)
np.save(os.path.join(MDIR, 'spgr_reconstr_hyperparams.npy'), hyperparams)
