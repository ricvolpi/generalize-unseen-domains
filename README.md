# Code for the paper [Generalizing to Unseen Domains via Adversarial Data Augmentation](https://arxiv.org/abs/1805.12018)

## Overview

### Files

``model.py``: to build tf's graph

``trainOps.py``: to train/test

``exp_configuration``: config file with the hyperparameters

### Prerequisites

Python 2.7, Tensorflow 1.6.0

## How it works

To obtain MNIST and SVHN dataset, run

```
python download_and_process_mnist.py
sh download_svhn.sh
```

To train the model, run

```
sh run_exp.sh
```

 
