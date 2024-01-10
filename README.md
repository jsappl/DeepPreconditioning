<div align="center">

# Deep Preconditioning

Design preconditioners with a CNN to accelerate the conjugate gradient method.

[![python](https://img.shields.io/badge/python-%3E%3D3.11-blue?logo=python)]()
[![pdm-managed](https://img.shields.io/badge/pdm-managed-blueviolet)](https://pdm.fming.dev)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

</div>

## Setup (Linux)

This has been tested with
* Ubuntu 18.04.4 LTS (GNU/Linux 4.15.0-112-generic x86_64)
* Python 3.6.9
* CUDA 11.0

Clone this repo using `git clone --recurse-submodules` which also pulls content from submodules. Initialize a virtual environment and install necessary dependencies with
```shell
virtualenv -p python3 env
. env/bin/activate
pip install -r requirements.txt
```

Compile the `spconv` package according to [these instructions](https://github.com/traveller59/spconv#install-on-ubuntu-16041804). No need to clone the repo itself since it is already included as a submodule. If you run into problems during the installation please refer to their issues section. We only use the `spconv` package as provided.

Install OpenFOAM 7 with `sudo apt install openfoam7`. We implemented a custom `interFoam` solver which dumps **L** and **d** from the discrete pressure Poisson equation **Lp=d** to disk. Change the directory `cd foam/newInterFoam/` and compile it by running `wmake`. Check out [https://openfoam.org/download/7-ubuntu/](https://openfoam.org/download/7-ubuntu/) if you have questions regarding the installation process of OpenFOAM 7 on Ubuntu.

# Training/testing the model
In the `preconditioner/` folder you can find [PyTorch](https://pytorch.org/) code for the machine learning part. Adjust the settings to your liking in `config.py`. Generate a data set of system matrices **L** representing the discretized Laplacian and start the train/test loop in the background with
```shell
python3 gen_data.py
nohup python3 train.py &
```
We use [TensorBoard](https://www.tensorflow.org/tensorboard/) to log hyperparameters, train/validation loss, and the performance of the model on the test data set. Run
```shell
tensorboard --logdir runs/ &
```
if you want to monitor the loss during training and check the test results in your browser.
