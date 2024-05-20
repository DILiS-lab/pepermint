# PEPerMINT
PEPerMINT (PEPtide Mass spectrometry Imputation NeTwork) is a graph-based deep neural network for the imputation of peptide abundance values in mass-spectrometry-based proteomics datasets.

## Installation
The implementation and evaluation of PEPerMINT are mostly based on our [PyProteoNet Python framework](https://github.com/Tobias314/pyproteonet), which is included in this repository as a submodule.

To install PEPerMINT, clone this repository together with the PyProteoNet submodule using the following command:

`git clone --recurse-submodules git@github.com:Tobias314/pepermint.git`

It is advisable to use Mamba to manage all dependencies in a clean environment (alternatively, you can, of course, use Conda by replacing all `mamba` commands with `conda` commands).
Within your environment, you need to install some dependencies required by PyProteoNet by running the following commands
- `mamba install -c conda-forge r-base` (R environment used to wrap and run some of the competitor imputation methods)
- `mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia` (PyTorch used as automatic differentiation and deep learning backend)
- `mamba install -c dglteam dgl` (for graph neural network functionality)
  
Once done, you can install PyProteoNet itself running the following command (you can remove the `-e` switch in case you do not want to have it editable):

`pip install -e pyprotoenet/.`

Last, but not least, it is advised to install Jupyter to run the evaluation and plotting code provided as Jupyter notebooks.

`mamba install -c conda-forge jupyterlab`


## Benchmarking PEPerMINT
We evaluate PEPerMINT on multiple benchmark datasets. To rerun our analysis, follow the steps described below.

### Data
While most of the datasets can be downloaded from [PRIDE](https://www.ebi.ac.uk/pride/), we provide a [Zenodo repository](https://zenodo.org/records/11216899) containing just the required abundance files in an easy-to-read format.
Just download the `data.zip` file, extract the `datasets/` folder, and place it inside the `data/datasets` folder of your cloned repository. Alternatively, you can also download the already preprocessed or imputed datasets from the same link by extracting the `datasets_preprocessed` and `datasets_imputed` folders from the `data.zip` file followed by placing them inside the `data` folder of your repository.

### Preprocessing
For benchmarking purposes, we introduce artificial missing values into some of the datasets.
In addition, we logarithmize all datasets, remove peptides showing only missing values and proteins without any assigned peptides, and sum-normalize the ratio datasets. 
For our neural network imputation, we also need to generate sequence embeddings for all the peptide sequences in every dataset using a pre-trained ProtT5 transformer model.
All this can be done by running 

`python preprocessing.py`

This loads the datasets, runs the preprocessing, and saves the preprocessed datasets to `data/datasets_preprocessed`. Generated sequence embeddings are cached.
To speed up the embedding generation on the first run, you might consider downloading already generated and cached embeddings from Zenodo. To do so download the [sequence_embedding_cache.h5](https://zenodo.org/records/11216899/files/sequence_embedding_cache.h5?download=1) file and place it inside the root repository folder. 

### Imputation
To run both the competitor imputation methods and our newly proposed deep neural network for imputation on the preprocessed datasets, just run `imputation.py` (running all imputation methods on all datasets might take several hours and should be done on a system with enough GPU memory for neural network training and prediction).

`python imputation.py`

### Evaluation
Evaluation and plotting functionality are provided inside the `evaluation.ipynb` Jupyter notebook.
