# GP
Gaussian processes (GP) for microscopic data and measurements. To use it, first run:

```
git clone https://github.com/ziatdinovmax/GP.git
cd GP
pip install -r req.txt
```

To perform GP-based reconstruction of sparse 2D image or hyperspectral 3D data, run:
```
python3 reconstruct.py <path/to/file.npy>
```

To perform GP-guided sample exploration with hyperspectral (3D) measurements based on the reduction of maximal uncertainty, run: 
```
python3 explore.py <path/to/file.npy>
```
Notice that the exploration part currently runs only "synthetic experiments" where you need to provide a full dataset (no missing values) as ground truth.

See also our executable Googe Colab [notebook](https://colab.research.google.com/github/ziatdinovmax/GP/blob/master/notebooks/GP_BEPFM.ipynb) with example of applying GP to both hyperspectral data reconstruction and sample exploration.

It is strongly recommended to run the codes with a GPU hardware accelerator. If you don't have a GPU on your local machine, you may rent a cloud GPU from a [Google Cloud AI Platform](https://cloud.google.com/ai-platform/). Running the [example notebook](https://colab.research.google.com/github/ziatdinovmax/GP/blob/master/notebooks/GP_BEPFM.ipynb) one time from top to bottom will cost about 2 USD with a standard deep learning VM instance (One P100 GPU and 15 GB of RAM).

More details TBA
