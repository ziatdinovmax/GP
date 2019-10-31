# GP
Gaussian processes (GP) for microscopic measurements (so far running "synthetic experiments").

To perform GP-based reconstruction of sparse 2D image or hyperspectral 3D data run
```
python3 reconstruct.py <path/to/file.npy>
```

To perform GP-based sample exploration with hyperspectral (3D) measurements based on the reduction of maximal uncertainty run 
```
python3 explore.py <path/to/file.npy>
```

See also [our notebook](https://colab.research.google.com/github/ziatdinovmax/GP/blob/master/notebooks/GP_BEPFM.ipynb) with example of applying GP to both hyperspectral data reconstruction and sample exploration.

It is strongly recommended to run the codes with GPU hardware accelerator.

More details TBA
