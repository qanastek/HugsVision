#!/usr/bin/env python3
import os
import setuptools
from distutils.core import setup

with open("README.md") as f:
    long_description = f.read()

with open(os.path.join("hugsvision", "version.txt")) as f:
    version = f.read().strip()

setup(
    name = "hugsvision",
    version = version,
    description = "A easy to use huggingface wrapper for computer vision.",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    author = "Yanis Labrak & Others",
    author_email = "yanis.labrak@univ-avignon.fr",
    packages = setuptools.find_packages(),
    package_data = {
        "hugsvision": [
            "version.txt"
        ]
    },
    install_requires = [
        "torch",
        "torchvision",
        "torchmetrics",
        "Pillow",
        "scikit-learn",
        "transformers",
        "tqdm",
        "tabulate",
        "timm",
        "matplotlib",
        "opencv-python",
        "pytorch-lightning",
        "pycocotools",
    ],
    python_requires = ">=3.6",
    url = "https://HugsVision.github.io/",
    keywords = ["python","transformers","huggingface","wrapper","toolkit","computer vision","easy","computer","vision"],
)