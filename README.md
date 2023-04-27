![icarogw2.0](/docs/logo.png)

# icarogw 2.0 Developers branch

## Installation

* Clone this repository
* Run the command `conda env create -f env_creator.yml ` if you are on linux or   `conda env create -f env_creator_mac.yml ` if you are on MAC OS  to install the package with conda.
* Run `conda activate icarogw2.0` to switch to the environment.

It is possible that you machine does not support CUDA. In this case, the code will not install CUDA and CUPY but should still be able to run. Please let me know if you encouter any difficulty and the installation is broken.

## Some details

When you are using icarogw2.0, you can put inside your working folder a file called `config.py` containing

```latex
CUPY=False
```

if you do not want to use the GPU. IF you want to use the GPU, remember to use cupy instead of numpy

## Example and Notebooks

There is a notebook called `stable_tests.ipynb`, where you will find how to call many commands. We also have [this repository](https://git.ligo.org/simone.mastrogiovanni/icarogw_catalog_tests) with additional notebooks, where we are collecting tests for the review. There is also a [living presentation](https://docs.google.com/presentation/d/14OgAo1Uj7NvnIGRfVTWMYJ7JDebn9Ns5ex6EIGPlbj4/edit?usp=sharing) where we are collecting some details and commands of the code.
