#!/bin/bash
python main.py --gpu=$1 --mode=train --exp_dir=./
python main.py --gpu=$1 --mode=test --exp_dir=./



