# GP
Gaussian processes (GP) for microscopic data analysis and measurements based on Pyro probabilistic programming language. To use it, first run:

```
git clone https://github.com/ziatdinovmax/GP.git
cd GP
python3 -m pip install -r req.txt
```

To perform GP-based reconstruction of sparse 2D image or sparse hyperspectral 3D data (datacube where measurements (spectroscopic curves) are missing for various xy positions), run:
```
python3 reconstruct.py <path/to/file.npy>
```
The missing values in the sparse data must be [NaNs](https://docs.scipy.org/doc/numpy/reference/constants.html?highlight=numpy%20nan#numpy.nan). If the data provided doesn't have missing values, it will be interpreted as a ground truth and a sparse copy of this dataset will be created. You can control the sparsity by passing ```--PROB``` argument (use ```python3 reconstruct.py -h``` to see other optional arguments). The ```reconstruct.py``` will return a zipped archive (.npz format) of numpy files corresponding to the ground truth (if applicable), input data, predictive mean and variance, and learned kernel hyperparameters. You can use ```python3 plot.py <path/to/file.npz>``` to view the results.

To perform GP-guided sample exploration with hyperspectral (3D) measurements based on the reduction of maximal uncertainty, run: 
```
python3 explore.py <path/to/file.npy>
```
Notice that the exploration part currently runs only "synthetic experiments" where you need to provide a full dataset (no missing values) as a ground truth.

See also our executable Googe Colab [notebook](https://colab.research.google.com/github/ziatdinovmax/GP/blob/master/notebooks/GP_BEPFM.ipynb) with examples of applying GP to both hyperspectral data reconstruction and sample exploration.

It is strongly recommended to run the codes with a GPU hardware accelerator. If you don't have a GPU on your local machine, you may rent a cloud GPU from [Google Cloud AI Platform](https://cloud.google.com/deep-learning-vm/). Running the [example notebook](https://colab.research.google.com/github/ziatdinovmax/GP/blob/master/notebooks/GP_BEPFM.ipynb) one time from top to bottom will cost about 1 USD with a standard deep learning VM instance (one P100 GPU and 15 GB of RAM).

More details TBA
