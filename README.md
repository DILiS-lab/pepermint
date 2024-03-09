# PEPerMINT
PEPerMINT (PEPtide Mass spectrometry Imputation NeTwork)) is a graph-based deep neural network for the imputation of peptide abundance values in mass-spectrometry-based proteomics datasets.

## Installation
The implementation as well as evaluation of PEPerMINT is mostly based on our [PyProteoNet Python framework](https://github.com/Tobias314/pyproteonet) which is included into this repository as submodule.

To install PEPerMINT first clone this repository together with the PyProteoNet submodule using the following command:

`git clone --recurse-submodules git@github.com:Tobias314/pepermint.git`

It is advisable to use Mamba to have a clean environment managing all dependencies (alternatively you can of course use Conda by just replacing all `mamba` with `conda` commands).
Within your environment, first install some dependencies required by PyProteoNet by running the following commands
- `mamba install -c conda-forge r-base` (R environment used to wrap and run some of the competitor imputation methods)
- `mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia` (PyTorch used as automatic differentiation and deep learning backend)
- `mamba install -c dglteam dgl` (for graph neural network functionanily)
  
Once done you can install PyProteoNet itself running the following command (you can remove the `-e` switch in case you do not want to have it editable):

`pip install -e pyprotoenet/.`

Last, but not least, it is advised to install Jupyter to run the evaluation and plotting code provided as Jupyter notebooks.

`mamba install -c conda-forge jupyterlab`


## Benchmarking PEPerMINT
We evaluate PEPerMINT on multiple benchmark datasets. To rerun our analysis follow the steps described below.

### Data
While most of the datasets can be downloaded from [PRIDE](https://www.ebi.ac.uk/pride/) we provide a Google Drive folder containing just the required (and much smaller) abundance files in an easy to read format.
Just use this [link](https://drive.google.com/drive/folders/1_YDJqfC5THMlJwJUsD-6O9tWdfzsacUL?usp=sharing), download the `datasets/` folder and place it inside the `data/datasets` folder of your cloned repository. Alternatively, you can also download the already preprocessed or imputed datasets from the same link by downloading the `datasets_preprocessed` and `datasets_imputed` followed by placing them inside the `data` folder of your repository.

### Preprocessing
For benchmarking puprpose we introduce artificial missing values into some of the datasets.
In, addition we logarithmize all datasets, remove peptides showing only missing values as well as proteins without any assigned peptides and sum normalize the ratio datasets. 
For our neural network imputation we also need to generate sequence embeddings for all the peptide sequences in every dataset using a pretrained ProtT5 transformer model.
All this can be done by running 

`python preprocessing.py`

This loads the datasets, runs the preprocessing and saves the preprocessed datasets to `data/datasets_preprocessed`. Generated sequence embeddings are cached.
To speed up the embedding generation on the first run you might consider downloading [this file](https://drive.google.com/file/d/1uEtNgq_sAdE24rp-X7c4CXeCeaEel1Aa/view?usp=sharing) containing the already cached embeddings and place it inside the root repository folder. 

### Imputation
To run both the competitor imputation methods as well as our newly proposed deep neural network for imputation on the preprocessed datasets just run `imputation.py` (running all imputation methods on all datasets might take several hours and should be done on a system with enough GPU memory for neural network training and prediction.).

`python imputation.py`

### Evaluation
Evaluation as well as plotting functionality is provided inside the `evaluation.ipynb` Jupyter notebook.
