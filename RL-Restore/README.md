## RL-Restore [[project page](http://mmlab.ie.cuhk.edu.hk/projects/RL-Restore/)][[paper](https://arxiv.org/abs/1804.03312)]

Original work by Ke Yu et al. This repository contains our adaptation of RL-Restore for astronomical image restoration.

### Overview

This project adapts the RL-Restore framework to specifically handle astronomical image degradation and restoration. We focus on addressing common issues in astronomical imaging such as point spread function (PSF) effects, readout noise, and maintaining high dynamic range.

### Prerequisites

- Python 3.6+
- Required packages can be installed via: `pip install -r requirements.txt`


### Dataset Setup

The DeepSpaceYoloDataset containing training images should be placed in the `data/` folder:

1. Navigate to the data/train directory:

2. Be in the data/train directory and run : python data/train/generate_train.py \
  --input_dir data/DeepSpaceYoloDataset \
  --train_output data/train/star_train.h5 \
  --val_output data/valid/validation.h5 \
  --val_split 0.2 \
  --max_val_images 100 \
  --max_train_images 500

3. Navigate back to RL-Restore directory and run:  python main.py --is_train True to start the training process



