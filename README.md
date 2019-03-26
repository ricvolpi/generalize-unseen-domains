# Code for the paper [Generalizing to Unseen Domains via Adversarial Data Augmentation](https://arxiv.org/abs/1805.12018)

## Overview

### Files

``model.py``: to build tf's graph

``trainOps.py``: to train/test

``exp_configuration``: config file with the hyperparameters

### Prerequisites

Python 2.7, Tensorflow 1.12.0

## How it works

To obtain MNIST and SVHN dataset, run

```
mkdir data
python download_and_process_mnist.py
sh download_svhn.sh
```
##
To train the model, run

```
python main.py --mode=train_MODE --gpu=GPU_IDX -- exp_dir=EXP_DIR
```
where MODE can be one of {ERM, RDA, RSDA, ESDA}, GPU_IDX is the index of the GPU to be used, and EXP_DIR is the folder containing the exp_config file.

##
To run evolution search (ES) or random search (RS) on a trained model, run

```
python main.py --mode=test_MODE --gpu=GPU_IDX -- exp_dir=EXP_DIR
```
where MODE can be one of {RS, ES}.
 
##
To test performance on all digit datasets (MNIST, SVHN, MNIST-M, SYN, USPS), run

```
python main.py --mode=test_all --gpu=GPU_IDX -- exp_dir=EXP_DIR
```

##




