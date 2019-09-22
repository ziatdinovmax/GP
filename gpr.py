# GPR model training and prediction
# Author: Maxim Ziatdinov (email: maxim.ziatdinov@ai4microcopy.com)

import time
import numpy as np
import torch
import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist
import gprutils


def get_kernel(kernel_type='RBF', input_dim=3, on_gpu=True, **kwargs):
    """
    Initalizes one of the following kernels:
    RBF, Rational Quadratic, Matern, Periodic kernel

    Args:
        kernel_type: str
            kernel type ('RBF', 'Rational Quadratic', 'Periodic', Matern52')
        input_dim: int
            number of input dimensions
            (equal to number of feature vector columns)
        on_gpu: bool
            sets default tensor type to torch.cuda.DoubleTensor

    **Kwargs:
        lengthscale: list of two lists
            determines lower (1st list) and upper (2nd list) bounds
            for kernel lengthscale(s) (default is from 0.1 to 20);
            number of elements in each list is equal to the input dimensions
        amplitude: list with two floats
            determines bounds on kernel amplitude parameter
            (default is from 1e-4 to 10)
        len_dim: int
            number of lengthscale tensor dimensions
            (allows using only one lengthscale for >1D feature vector)

    Returns:
        kernel object
    """

    if on_gpu:
        torch.set_default_tensor_type(torch.cuda.DoubleTensor)

    amp = kwargs.get('amplitude') if 'amplitude' in kwargs else [1e-4, 10.]
    len_dim = kwargs.get('len_dim') if 'len_dim' in kwargs else input_dim
    lscale = kwargs.get('lengthscale')
    if lscale is None:
        lscale = [[.1 for l in range(len_dim)], [20. for l in range(len_dim)]]
    lscale_ = torch.tensor(lscale[0]) + 1e-5

    # initialize the kernel
    kernel_book = lambda input_dim, len_dim: {
        'RBF': gp.kernels.RBF(
            input_dim, lengthscale=lscale_
            ),
        'RationalQuadratic': gp.kernels.RationalQuadratic(
            input_dim, lengthscale=lscale_
            ),
        'Matern52': gp.kernels.Matern52(
            input_dim, lengthscale=lscale_)
    }

    try:
        kernel = kernel_book(input_dim, len_dim)[kernel_type]
    except KeyError:
        print('Select one of the currently available kernels:',\
              '"RBF", "RationalQuadratic", "Matern52"')
        raise

    # set priors
    kernel.set_prior(
        "variance",
        dist.Uniform(
            torch.tensor(amp[0]),
            torch.tensor(amp[1])
        )
    )
    kernel.set_prior(
        "lengthscale",
        dist.Uniform(
            torch.tensor(lscale[0]),
            torch.tensor(lscale[1])
        ).to_event()
    )

    return kernel


def train_sgpr_model(X, y, kernel, learning_rate=5e-2, steps=1000,
                    use_gpu=True, verbose=False):
    """
    Training sparse GP regression model

    Args:
        X: ndarray with N x c dimensions
            c is equal to the number of coordinate dimensions.
            For example, for xyz coordinates, c = 3.
            N is equal to number of observations (data points)
        y: ndarray with (N,) dimensions
            Observations (data points)
        kernel: Pyro kernel object
            Initialized kernel
        learning_rate: float
            learning rate
        steps: int
            number of SVI training iteratons
        use_gpu: bool
            Uses GPU hardware accelerator when set to 'True'
        verbose: bool
            prints statistics after each 100th training iteration

    Returns:
        Trained model (GPregression object), list of training losses
    """

    pyro.set_rng_seed(0)
    pyro.clear_param_store()
    # initialize the inducing inputs
    indpoints = int(len(X)*1e-2)
    indpoints = 150 if indpoints > 150 else indpoints
    indpoints = 20 if indpoints == 0 else indpoints
    Xu = X[::len(X) // indpoints]
    print("Number of inducing points: {}".format(len(Xu)))
    if use_gpu:
        X = X.cuda()
        y = y.cuda()
    # initialize the model
    sgpr = gp.models.SparseGPRegression(X, y, kernel, Xu=Xu, jitter=1.0e-5)
    # learn the model parameters
    if use_gpu:
        sgpr.cuda()
    optimizer = torch.optim.Adam(sgpr.parameters(), lr=learning_rate)
    loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
    losses = []
    num_steps = steps
    start_time = time.time()
    for i in range(num_steps):
        optimizer.zero_grad()
        loss = loss_fn(sgpr.model, sgpr.guide)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if verbose and (i % 100 == 0 or i == num_steps - 1):
            print('iter: {} ...'.format(i),
                  'loss: {} ...'.format(np.around(losses[-1], 4)),
                  'amp: {} ...'.format(
                    np.around(sgpr.kernel.variance.item(), 4)),
                  'length: {} ...'.format(
                    np.around(sgpr.kernel.lengthscale.tolist(), 4)),
                  'noise: {} ...'.format(np.around(sgpr.noise.item(), 7)))
        if i == 100:
            print('average time per iteration: {} s'.format(
                np.round(time.time() - start_time, 2) / 100))
    print('training completed in {} s'.format(
        np.round(time.time() - start_time, 2)))
    print('Final parameter values:\n',
          'amp: {}, lengthscale: {}, noise: {}'.format(
              np.around(sgpr.kernel.variance.item(), 4),
              np.around(sgpr.kernel.lengthscale.tolist(), 4),
              np.around(sgpr.noise.item(), 7)
          ))
    if use_gpu:
        X = X.cpu()
        y = y.cpu()
        sgpr.cpu()  # this actually doesn't seem to work in Pyro as intented

    return sgpr, losses

def sgpr_predict(model, Xtest, use_gpu=True, num_batches=10):
    """
    Use trained GPRegression model to make predictions

    Args:
        model: GPRegression object
            Trained GP regression model
        Xtest: N x c ndarray
            "Test" coordinate indices
        use_gpu: bool
            Uses GPU hardware accelerator when set to 'True'
        num_batches: int
            number of batches for splitting the Xtest array
            (for large datasets, you may not have enough GPU memory
            to process the entire dataset at once)

    Returns:
        predictive mean and variance
    """

    # Prepare for inference
    Xtest = prepare_test_data(Xtest)
    batch_range = len(Xtest) // num_batches
    mean = np.zeros((Xtest.shape[0]))
    sd = np.zeros((Xtest.shape[0]))
    # Run inference batch-by-batch
    for i in range(num_batches):
        Xtest_i = torch.from_numpy(Xtest[i*batch_range:(i+1)*batch_range])
        if use_gpu:
            Xtest_i = Xtest_i.cuda()
            model = model.cuda()
        with torch.no_grad():
            mean_i, cov = model(Xtest_i, full_cov=True, noiseless=False)
        sd_i = cov.diag().sqrt()
        mean[i*batch_range:(i+1)*batch_range] = mean_i.cpu().numpy()
        sd[i*batch_range:(i+1)*batch_range] = sd_i.cpu().numpy()
    if use_gpu:
        model.cpu()

    return mean, sd


def exploration_step(X, R, X_true, dist_edge, learning_rate=.1):
    """
    Finds new point with maximum uncertainty

    Args:
        X:  c x  N x M x L ndarray
            Grid indices for initial partial observations
            c is equal to the number of coordinate dimensions.
            For example, for xyz coordinates, c = 3.
            Missing points are NaNs.
        R: N x M x L ndarray
            Observations (data points).
            Missing data points are NaNs.
        X_true:  c x  N x M x L ndarray
            Grid indices for full observations (i.e. without NaNs)
            The dimensions are equal to those of X
        dist_edge: list with two integers
            edge regions not considered for max uncertainty evaluation
        learning_rate: float
            learning rate for GPR model training

    Returns:
        list indices of point with maximum uncertainty

    """
    e1, e2, e3 = R.shape
    # pre-process data
    X_tor, R_tor = gprutils.prepare_training_data(X, R)
    # use different bounds on lengthscale at the very beginning
    lscale = None if i < 10 else [[4., 4., 2.5], [5., 5., 5.]]
    # train a model
    model, losses = train_sgpr_model(
        X_tor, R_tor, get_kernel(
            'RationalQuadratic', len_dim=1, lengthscale=lscale),
        learning_rate=learning_rate, steps=1500, use_gpu=1, verbose=1
        )
    # make prediction
    mean, sd = sgpr_predict(
        model, gprutils.prepare_test_data(X_true),
        use_gpu=False, num_batches=100
        )
    # find point with maximum uncertainty
    sd = sd.reshape(e1, e2, e3)
    amax = gprutils.max_uncertainty(sd, dist_edge)

    return amax, mean, sd