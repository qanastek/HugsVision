# HugsVision

<p align="center">
  <img src="https://github.com/qanastek/HugsVision/blob/main/ressources/images/logo_name_transparent.png" alt="drawing" width="250"/>
</p>

HugsVision is a open-source and easy to use all-in-one huggingface wrapper for computer vision.

The goal is to create a fast, flexible and user-friendly toolkit than can be used to easily develop **state-of-the-art** computer vision technologies, including systems for Image Classification, Semantic Segmentation, Object Detection, Image Generation, Denoising and much more.

⚠️ HugsVision is currently in beta. ⚠️

# Quick installation

HugsVision is constantly evolving. New features, tutorials, and documentation will appear over time. HugsVision can be installed via PyPI to rapidly use the standard library. Moreover, a local installation can be used by those users that what to run experiments and modify/customize the toolkit. HugsVision supports both CPU and GPU computations. For most all the recipes, however, a GPU is necessary during training. Please note that CUDA must be properly installed to use GPUs.

## Anaconda setup

Once Anaconda is installed, you can create a new environment with:

```bash
conda create --name HugsVision python=3.6 -y
```

Then, activate it with:

```bash
conda activate HugsVision
```

Now, you can install all the needed packages!

More information on managing environments with Anaconda can be found in [the conda cheat sheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf).

## Install via PyPI

Once you have created your Python environment (Python 3.6+) you can simply type:

```bash
pip install hugsvision
```

Then you can access HugsVision with:

```python
import hugsvision as vision
```

## Install with GitHub

Once you have created your Python environment (Python 3.6+) you can simply type:

```bash
git clone https://github.com/qanastek/HugsVision.git
cd HugsVision
pip install -r requirements.txt
pip install --editable .
```

Then you can access HugsVision with:

```python
import hugsvision as vision
```

Any modification made to the `hugsvision` package will be automatically interpreted as we installed it with the `--editable` flag.

# Running an experiment

In HugsVision, you can run experiments in this way:

```bash
cd recipes/<DATASET>/<TASK>/
python train_example_vit.py --imgs="<PATH_TO_IMAGES_DIRECTORY>" --name="<OUTPUT_MODEL_NAME>"
python predict.py --img="<IMG_PATH>" --path="<MODEL_PATH>"
```

For example, we can use the `Pneumothorax` dataset available on [Kaggle](https://www.kaggle.com/volodymyrgavrysh/pneumothorax-binary-classification-task) to train a binary classification model.

Steps:

- Move to the recipe directory `cd recipes/pneumothorax/binary_classification/`
- Download the dataset [here](https://www.kaggle.com/volodymyrgavrysh/pneumothorax-binary-classification-task) ~779 MB.
- Transform the dataset into a directory based one, thanks to the `process.py` script.
- Train the model
  - `python train_example_vit.py --imgs="./pneumothorax_binary_classification_task_data/" --name="pneumo_model_vit" --epochs=1`
- Rename `<MODEL_PATH>/config.json` to `<MODEL_PATH>/preprocessor_config.json` in my case, the model is situated at the path `./out/MYVITMODEL/1_2021-08-10-00-53-58/model/`
- Make a prediction
  - `python predict.py --img="42.png" --path="./out/MYVITMODEL/1_2021-08-10-00-53-58/model/"`

# Model architectures

All the model checkpoints provided by 🤗 Transformers and compatible with our tasks can be seamlessly integrated from the huggingface.co model hub where they are uploaded directly by users and organizations.

Before starting implementing, please check if your model has an implementation in `PyTorch` by refering to [this table](https://huggingface.co/transformers/index.html#supported-frameworks).

🤗 Transformers currently provides the following architectures for Computer Vision:

1. **[ViT](https://huggingface.co/transformers/model_doc/vit.html)** (from Google Research, Brain Team) released with the paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/pdf/2010.11929.pdf), by Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby.
2. **[DeiT](https://huggingface.co/transformers/model_doc/deit.html)** (from Facebook AI and Sorbonne University) released with the paper [Training data-efficient image transformers & distillation through attention](https://arxiv.org/pdf/2012.12877.pdf) by Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, Hervé Jégou.
3. **[DETR](https://huggingface.co/transformers/model_doc/detr.html)** (from Facebook AI) released with the paper [End-to-End Object Detection with Transformers](https://arxiv.org/pdf/2005.12872.pdf) by Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov and Sergey Zagoruyko.

# Build PyPi package

Build: `python setup.py sdist bdist_wheel`

Upload: `twine upload dist/*`
