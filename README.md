![icarogw2.0](/docs/logo.png)

# icarogw 2.0 Developers branch

## Installation

Simply run

```
pip install git+https://github.com/simone-mastrogiovanni/icarogw.git
```

To install the latest version of icarogw


### Installing the GPU CUDA version

To use the GPU/CUDA version you need to install cupy. After your installation is complete, just run

```
conda install -c conda-forge cupy==12.0
```

## Some details

When you are using icarogw2.0, you can put inside your working folder a file called `config.py` containing

```latex
CUPY=False
```

if you do not want to use the GPU. IF you want to use the GPU, remember to use cupy instead of numpy

## Example and Notebooks

There is a notebook called `stable_tests.ipynb`, where you will find how to call many commands. We also have [this repository](https://git.ligo.org/simone.mastrogiovanni/icarogw_catalog_tests) with additional notebooks, where we are collecting tests for the review. There is also a [living presentation](https://docs.google.com/presentation/d/14OgAo1Uj7NvnIGRfVTWMYJ7JDebn9Ns5ex6EIGPlbj4/edit?usp=sharing) where we are collecting some details and commands of the code.
