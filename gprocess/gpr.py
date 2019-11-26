'''
(Sparse) Gaussian process regression: model training, prediction and uncertainty exploration

Serves as a high-level wrapper for Gaussian processes module 
from Pyro probabilistic programming library (https://pyro.ai/)
for easy work with scientific image (2D) and hyperspectral (3D, 4D) data.

Author: Maxim Ziatdinov (email: maxim.ziatdinov@ai4microcopy.com)
'''

import time
import numpy as np
import gprocess.gprutils as gprutils
import torch
import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist
import warnings


class reconstructor:
    """
    Class for Gaussian process regression-based reconstuction
    of sparse 2D image and 3D spectroscopic datasets, 
    and sample exploration with hyperspectral measurements
    based on maximal uncertainty reduction

    Args:
        X:  c x  N x M x L or c x N x M ndarray
            Grid indices.
            c is equal to the number of coordinate dimensions.
            For example, for xyz coordinates, c = 3.
        y: N x M x L or N x M ndarray
            Observations (data points)
        X_test: N x M x L or N x M ndarray
            "Test" points (for prediction with a trained GP model)
        kernel: str
            kernel type
        lengthscale: list of two lists
            determines lower (1st list) and upper (2nd list) bounds
            for kernel lengthscale(s)
        indpoints: int
            number of inducing points for SparseGPRegression
        input_dim: int
            number of lengthscale dimensions (1 or 3)
        learning_rate: float
            learning rate
        iterations: int
            number of SVI training iteratons
        num_batches: int
            number of batches for splitting the Xtest array
            (for large datasets, you may not have enough GPU memory
            to process the entire dataset at once)
        use_gpu: bool
            Uses GPU hardware accelerator when set to 'True'
        verbose: bool
            prints statistics after each 100th training iteration
        
    **Kwargs:
        amplitude: kernel variance or amplitude squared

    Methods:
        train:
            Training sparse GP regression model
        predict:
            Using trained GP regression model to make predictions
        run:
            Combines training and prediction to output mean, SD
            and hyperaprameters as a function of SVI steps
        step:
            Combines a single model training and prediction
            to find point with max uncertainty in the data
    """
    def __init__(self, X, y, Xtest,
                 kernel, lengthscale=None,
                 indpoints=1000, input_dim=3,
                 learning_rate=5e-2, iterations=1000,
                 num_batches=10, use_gpu=False, 
                 verbose=False, **kwargs):
        if use_gpu and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.set_default_tensor_type(torch.cuda.DoubleTensor)
            use_gpu = True
        else:
            torch.set_default_tensor_type(torch.DoubleTensor)
            use_gpu = False
        self.X, self.y = gprutils.prepare_training_data(X, y)
        if indpoints > len(self.X):
            indpoints = len(self.X)
        Xu = self.X[::len(self.X) // indpoints]
        if lengthscale is None:
            lengthscale = [[0. for l in range(input_dim)], 
                           [np.mean(y.shape) / 2 for l in range(input_dim)]]
        kernel = get_kernel(kernel, input_dim, 
                            lengthscale, use_gpu, 
                            amplitude=kwargs.get('amplitude'))
        self.fulldims = Xtest.shape[1:]
        self.Xtest = gprutils.prepare_test_data(Xtest)
        if use_gpu:
            self.X = self.X.cuda()
            self.y = self.y.cuda()
            self.Xtest = self.Xtest.cuda()
        self.sgpr = gp.models.SparseGPRegression(
            self.X, self.y, kernel, Xu, jitter=1.0e-5)
        print("# of inducing points for GP regression: {}".format(len(Xu)))
        if use_gpu:
            self.sgpr.cuda()
        self.num_batches = num_batches
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.hyperparams = {}
        self.indpoints_all = []
        self.lscales, self.noise_all, self.amp_all = [], [], []
        self.hyperparams = {
            "lengthscale": self.lscales,
            "noise": self.noise_all,
            "variance": self.amp_all,
            "inducing_points": self.indpoints_all
        }
        self.verbose = verbose

    def train(self, **kwargs):
        """
        Training sparse GP regression model
        **Kwargs:
            learning_rate: float
                learning rate
            iterations: int
                number of SVI training iteratons
        """
        if kwargs.get("learning_rate") is not None:
            self.learning_rate = kwargs.get("learning_rate")
        if kwargs.get("iterations") is not None:
            self.iterations = kwargs.get("iterations")
        pyro.set_rng_seed(0)
        pyro.clear_param_store()
        optimizer = torch.optim.Adam(self.sgpr.parameters(), lr=self.learning_rate)
        loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
        num_steps = self.iterations
        start_time = time.time()
        print('Model training...')
        for i in range(self.iterations):
            optimizer.zero_grad()
            loss = loss_fn(self.sgpr.model, self.sgpr.guide)
            loss.backward()
            optimizer.step()
            self.lscales.append(self.sgpr.kernel.lengthscale_map.tolist())
            self.amp_all.append(self.sgpr.kernel.variance_map.item())
            self.noise_all.append(self.sgpr.noise.item())
            self.indpoints_all.append(self.sgpr.Xu.detach().cpu().numpy())
            if self.verbose and (i % 100 == 0 or i == self.iterations - 1):
                print('iter: {} ...'.format(i),
                      'loss: {} ...'.format(np.around(loss.item(), 4)),
                      'amp: {} ...'.format(np.around(self.amp_all[-1], 4)),
                      'length: {} ...'.format(np.around(self.lscales[-1], 4)),
                      'noise: {} ...'.format(np.around(self.noise_all[-1], 7)))
            if i == 100:
                print('average time per iteration: {} s'.format(
                    np.round(time.time() - start_time, 2) / 100))
        print('training completed in {} s'.format(
            np.round(time.time() - start_time, 2)))
        print('Final parameter values:\n',
              'amp: {}, lengthscale: {}, noise: {}'.format(
                np.around(self.sgpr.kernel.variance_map.item(), 4),
                np.around(self.sgpr.kernel.lengthscale_map.tolist(), 4),
                np.around(self.sgpr.noise.item(), 7)))
        return

    def predict(self, **kwargs):
        """
        Use trained GPRegression model to make predictions
        **Kwargs:
            num_batches: int
                number of batches for splitting the Xtest array
                (for large datasets, you may not have enough GPU memory
                to process the entire dataset at once)
        Returns:
            predictive mean and variance
        """
        if kwargs.get("num_batches") is not None:
            self.num_batches = kwargs.get("num_batches")
        print("Calculating predictive mean and variance...", end=" ")
        # Prepare for inference
        batch_range = len(self.Xtest) // self.num_batches
        mean = np.zeros((self.Xtest.shape[0]))
        sd = np.zeros((self.Xtest.shape[0]))
        # Run inference batch-by-batch
        for i in range(self.num_batches):
            Xtest_i = self.Xtest[i*batch_range:(i+1)*batch_range]
            with torch.no_grad():
                mean_i, cov = self.sgpr(Xtest_i, full_cov=True, noiseless=False)
            sd_i = cov.diag().sqrt()
            mean[i*batch_range:(i+1)*batch_range] = mean_i.cpu().numpy()
            sd[i*batch_range:(i+1)*batch_range] = sd_i.cpu().numpy()
        print("Done")
        return mean, sd

    def run(self, **kwargs):
        """
        Train the initialized model and calculate predictive mean and variance

        **Kwargs:
            learning_rate: float
                learning rate for GPR model training
            steps: int
                number of SVI training iteratons
            num_batches: int
                number of batches for splitting the Xtest array

        Returns:
            predictive mean and SD as flattened ndarrays
            dictionary with hyperparameters as a function of SVI steps

        """
        if kwargs.get("learning_rate") is not None:
            self.learning_rate = kwargs.get("learning_rate")
        if kwargs.get("iterations") is not None:
            self.iterations = kwargs.get("iterations")
        if kwargs.get("num_batches") is not None:
            self.num_batches = kwargs.get("num_batches")
        self.train(learning_rate=self.learning_rate, iterations=self.iterations)
        mean, sd = self.predict(num_batches=self.num_batches)
        if next(self.sgpr.parameters()).is_cuda:
            self.sgpr.cpu()
            torch.set_default_tensor_type(torch.DoubleTensor)
            self.X, self.y = self.X.cpu(), self.y.cpu()
            self.Xtest = self.Xtest.cpu()
            torch.cuda.empty_cache()
        return mean, sd, self.hyperparams

    def step(self, dist_edge, **kwargs):
        """
        Performs single train-predict step for exploration analysis
        returning a new point with maximum uncertainty

        Args:
            dist_edge: list with two integers
                edge regions not considered for max uncertainty evaluation

        **Kwargs:
            learning_rate: float
                learning rate for GPR model training
            steps: int
                number of SVI training iteratons
            num_batches: int
                number of batches for splitting the Xtest array

        Returns:
            list indices of point with maximum uncertainty

        """
        if kwargs.get("learning_rate") is not None:
            self.learning_rate = kwargs.get("learning_rate")
        if kwargs.get("iterations") is not None:
            self.iterations = kwargs.get("iterations")
        if kwargs.get("num_batches") is not None:
            self.num_batches = kwargs.get("num_batches")
        # train a model
        self.train(learning_rate=self.learning_rate, iterations=self.iterations)
        # make prediction
        mean, sd = self.predict(num_batches=self.num_batches)
        # find point with maximum uncertainty
        sd_ = sd.reshape(self.fulldims[0], self.fulldims[1], self.fulldims[2])
        amax, uncert_list = gprutils.max_uncertainty(sd_, dist_edge)
        return amax, uncert_list, mean, sd

    def train_sgpr_model(self, learning_rate=5e-2, steps=1000):
        print("Use reconstructor.run instead of reconstructor.train_sgpr_model",
        "and reconstructor.sgpr_predict to obtain mean, sd and hyperparameters")
        pass

    def sgpr_predict(self, model, num_batches=10):
        print("Use reconstructor.run instead of reconstructor.train_sgpr_model",
        "and reconstructor.sgpr_predict to obtain mean, sd and hyperparameters")
        pass


def get_kernel(kernel_type, input_dim, lengthscale, use_gpu=False, **kwargs):
    """
    Initalizes one of the following kernels:
    RBF, Rational Quadratic, Matern

    Args:
        kernel_type: str
            kernel type ('RBF', 'Rational Quadratic', Matern52')
        input_dim: int
            number of input dimensions
            (equal to number of feature vector columns)
        lengthscale: list of two lists
            determines lower (1st list) and upper (2nd list) bounds
            for kernel lengthscale(s).
            number of elements in each list is equal to the input dimensions
        use_gpu: bool
            sets default tensor type to torch.cuda.DoubleTensor

    **Kwargs:
        amplitude: list with two floats
            determines bounds on kernel amplitude parameter
            (default is from 1e-4 to 10)

    Returns:
        Pyro kernel object
    """
    if use_gpu and torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.DoubleTensor)
    else:
        torch.set_default_tensor_type(torch.DoubleTensor)

    amp = kwargs.get('amplitude')
    lscale = lengthscale
    amp = [1e-4, 10.] if amp is None else amp
    # Needed in Pyro < 1.0.0  
    lscale_ = torch.tensor(lscale[0]) + 1e-5

    # initialize the kernel
    kernel_book = lambda input_dim: {
        'RBF': gp.kernels.RBF(
            input_dim, lengthscale=lscale_
            ),
        'RationalQuadratic': gp.kernels.RationalQuadratic(
            input_dim, lengthscale=lscale_
            ),
        'Matern52': gp.kernels.Matern52(
            input_dim, lengthscale=lscale_
            )
    }

    try:
        kernel = kernel_book(input_dim)[kernel_type]
    except KeyError:
        print('Select one of the currently available kernels:',\
              '"RBF", "RationalQuadratic", "Matern52"')
        raise
        
    with warnings.catch_warnings():  # TODO: use PyroSample to set priors
        warnings.filterwarnings("ignore", category=UserWarning)
        
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
            ).independent()
        )

    return kernel


class explorer:
    def __init__(self, X, y, Xtest, kernel, lengthscale,
                 indpoints=1000, ldim=3, use_gpu=False, verbose=False):
        print("Use gpr.reconstructor instead of gpr.explorer")
        pass
