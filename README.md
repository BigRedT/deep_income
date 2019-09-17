# Deep Income

This repository is part of a guest lecture for [CS598RK: HCI for ML](https://courses.grainger.illinois.edu/cs598rk/fa2019/) offered in Fall 2019 at UIUC. The goal is to familiarize students with Pytorch and different components of Deep Learning through a Not-So-Big-Data ML problem -- learning to predict annual income using [UCI Census Income Dataset](https://archive.ics.uci.edu/ml/datasets/Adult) (a.k.a UCI Adult Dataset). 

## Installation
We will create a conda environment and install all dependencies in that environment. `environment.yml` lists all dependencies. If conda is already installed on your system the following command will create an environment called `income` (specified in the YAML file) and install dependencies in that environment:
```
conda env create -f environment.yml
```

Commands in the subsequent section need to be run with the environment activated. To activate the environment run
```
conda activate income
```

To deactivate run
```
conda deactivate
```

For more information on conda please refer to [conda docs](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html).

**Note:** The `environment.yml` file lists a lot of dependencies. But most of them are requirements of just a handful of packages that can be found in `install.sh`. This script also shows steps used to create the conda environment from scratch. Ideally executing this script using `bash install.sh` should produce a similar environment but may result in packages with different version numbers.

## Specify file paths
`income.yml` lists all global constants that will be used in the repository. These constants include:
- `urls`: URLs to download data from
- `download_dir`: where data will be downloaded to on your machine
- `proc_dir`: where preprocessed data will be saved
- `exp_dir`: where experiment data will be saved
- `train_val_split`: fraction of the provided train data to be used for training (the remaining will be used for validation)
- Names of various `.npy` files used for training/validation/testing. These are all saved in `proc_dir` and will be read by dataloaders.

You will need to modify `download_dir`,`proc_dir`, and `exp_dir` according to your own machine.

## Download data
```
python -m download
```
This will download the following files in your `download_dir`
- `adult.data`: Train samples
- `adult.test`: Test samples
- `adult.names`: Info on attributes and performance of several baselines
- `old.adult.names`: Possibly an older version of `adult.names`?

## Preprocess data
```
python -m preprocess
```

This converts the samples provided with 14 attributes (8 categorical and 6 continuous) into real valued vectors. Samples with missing values are dropped. Features are normalized by subtracting the mean and dividing by the standard deviation computed using training set samples. The training data is also divided into train and val sets.

## Training
```
python -m train
```

This trains a model with default arguments. To see the arguments and their default values run 
```
python -m train --help
```
which should show the following
```
Options:
  --exp_name TEXT               Name of the experiment  [default: default_exp]
  --loss [cross_entropy|focal]  Loss used for training  [default: cross_entropy]
  --num_hidden_blocks INTEGER   Number of hidden blocks in the classifier [default: 2]
```
Any outputs generated during training are saved in `proc_dir/exp_name`

To visualize loss and accuracy curves on tensorboard, go to the experiment directory and run 
```
tensorboard --logdir=./
```

# Testing

The model with the best validation performance during training can be loaded up and evaluated on the test set using
```
python -m test
```
Note that this would work only when default arguments were used during training. For training with non-default arguments use
```
python -m test --exp_name <experiment name> --num_hidden_blocks <number of hidden blocks in the classifier>
```

For default arguments the accuracies on various data subsets should be in the ballpark of the following

|Train|Val|Test|
|---|---|---|
|86.17|85.05|84.65|

**A note on reproducibility:** Reproducing the above numbers is possible only if all of the following are true:
- random seeds in the `preprocess.py` and `train.py` scripts are set to `0`
- `train_val_split` in `income.yml` is set to `0.8`
- default arguments are used during training