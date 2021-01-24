# Vision transformer examples - PyTorch

Based on https://github.com/lucidrains/vit-pytorch

## Dataset
 1. MNIST
 2. Cat and Dog (https://www.kaggle.com/c/dogs-vs-cats/data)

## Install

pip install -r requirements.txt <br>


## MNIST

Features: Hyper parameter tuning by optuna (https://github.com/optuna/optuna) on MNIST

python vit_mnist.py

<img src="evaluation.png" alt="Training curve accuracy" width="400" height="350">


## Cat and Dog

python vit_catanddog.py

<img src="vi_catanddog.png" alt="Training and validation curve accuracy" width="400" height="350">

