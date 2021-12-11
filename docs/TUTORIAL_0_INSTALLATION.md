
# Quick installation

HugsVision is constantly evolving. New features, tutorials, and documentation will appear over time. HugsVision can be installed via PyPI to rapidly use the standard library. Moreover, a local installation can be used to run experiments and modify/customize the toolkit.

HugsVision supports both CPU and GPU computations. For most recipes, however, a GPU is necessary during training. Please note that CUDA must be properly installed to use GPUs.

We recommands to use pytorch >= 1.9 (https://pytorch.org/), transformers >= 4.9 (https://github.com/huggingface/transformers) and Python >= 3.6.

## Install via PyPI

Once you have created your Python environment (see instructions below) you can simply type:

```bash
pip install HugsVision
```

Then you can then access HugsVision with:

```python
import HugsVision as vision
```

## Install locally

Once you have created your Python environment (see instructions below) you can simply type:

```bash
git clone https://github.com/qanastek/HugsVision.git
cd HugsVision
pip install -r requirements.txt
pip install --editable .
```

Then you can access HugsVision with:

```python
import HugsVision as vision
```

Any modification made to the `HugsVision` package will be automatically interpreted as we installed it with the `--editable` flag.

## Anaconda and virtual env

A good practice is to have different python environments for your different tools
and toolkits, so they do not interfere with each other. This can be done either with
[Anaconda](https://en.wikipedia.org/wiki/Anaconda_(Python_distribution)) or [virtual env](https://docs.python.org/3.6/library/venv.html).

Anaconda can be installed by simply following [this tutorial](https://docs.anaconda.com/anaconda/install/linux/). In practice, it is a matter of downloading the installation script and executing it.

## Anaconda setup

Once Anaconda is installed, you can create a new environment with:

```bash
conda create --name HugsVision python=3.6
```

Then, activate it with:

```bash
conda activate HugsVision
```

Now, you can install all the needed packages!

More information on managing environments with Anaconda can be found in [the conda cheat sheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf).

## Test installation

Please, run the following command to make sure your installation is working:

```bash
pip show hugsvision
```
