<p align="center">
  <img src="https://raw.githubusercontent.com/qanastek/HugsVision/main/ressources/images/logo_name_transparent.png" alt="drawing" width="250"/>
</p>

[![PyPI version](https://badge.fury.io/py/hugsvision.svg)](https://badge.fury.io/py/hugsvision)
[![GitHub Issues](https://img.shields.io/github/issues/qanastek/HugsVision.svg)](https://github.com/qanastek/HugsVision/issues)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://static.pepy.tech/personalized-badge/hugsvision?period=total&units=international_system&left_color=grey&right_color=orange&left_text=Downloads)](https://pepy.tech/project/hugsvision)

HugsVision is an open-source and easy to use all-in-one huggingface wrapper for computer vision.

The goal is to create a fast, flexible and user-friendly toolkit that can be used to easily develop **state-of-the-art** computer vision technologies, including systems for Image Classification, Semantic Segmentation, Object Detection, Image Generation, Denoising and much more.

⚠️ HugsVision is currently in beta. ⚠️

# Quick installation

HugsVision is constantly evolving. New features, tutorials, and documentation will appear over time. HugsVision can be installed via PyPI to rapidly use the standard library. Moreover, a local installation can be used by those users than want to run experiments and modify/customize the toolkit. HugsVision supports both CPU and GPU computations. For most recipes, however, a GPU is necessary during training. Please note that CUDA must be properly installed to use GPUs.

## Anaconda setup

```bash
conda create --name HugsVision python=3.6 -y
conda activate HugsVision
```

More information on managing environments with Anaconda can be found in [the conda cheat sheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf).

## Install via PyPI

Once you have created your Python environment (Python 3.6+) you can simply type:

```bash
pip install hugsvision
```

## Install with GitHub

Once you have created your Python environment (Python 3.6+) you can simply type:

```bash
git clone https://github.com/qanastek/HugsVision.git
cd HugsVision
pip install -r requirements.txt
pip install --editable .
```

Any modification made to the `hugsvision` package will be automatically interpreted as we installed it with the `--editable` flag.

# Example Usage

Let's train a binary classifier that can distinguish people with or without `Pneumothorax` thanks to their radiography.

**Steps:**

1. Move to the recipe directory `cd recipes/pneumothorax/binary_classification/`
2. Download the dataset [here](https://www.kaggle.com/volodymyrgavrysh/pneumothorax-binary-classification-task) ~779 MB.
3. Transform the dataset into a directory based one, thanks to the `process.py` script.
4. Train the model:  `python train_example_vit.py --imgs="./pneumothorax_binary_classification_task_data/" --name="pneumo_model_vit" --epochs=1`
5. Rename `<MODEL_PATH>/config.json` to `<MODEL_PATH>/preprocessor_config.json` in my case, the model is situated at the output path like `./out/MYVITMODEL/1_2021-08-10-00-53-58/model/`
6. Make a prediction: `python predict.py --img="42.png" --path="./out/MYVITMODEL/1_2021-08-10-00-53-58/model/"`

# Models recipes

You can find all the currently available models or tasks under the `recipes/` folder.

<table>
  <tr>
      <td rowspan="3" width="160">
        <img src="https://raw.githubusercontent.com/qanastek/HugsVision/main/ressources/images/receipes/pneumothorax.png" width="256">
      </td>    
      <td rowspan="3">
        <b>Training a Transformer Image Classifier to help radiologists detect Pneumothorax cases:</b> A demonstration of how to train a Image Classifier Transformer model that can distinguish people with or without Pneumothorax thanks to their radiography with HugsVision.
      </td>
      <td align="center" width="80">
          <a href="https://nbviewer.jupyter.org/github/qanastek/HugsVision/blob/main/recipes/pneumothorax/binary_classification/Image_Classifier.ipynb">
              <img src="https://raw.githubusercontent.com/qanastek/HugsVision/main/ressources/images/receipes/nbviewer_logo.svg" height="34">
          </a>
      </td>
  </tr>
  <tr>
      <td align="center">
          <a href="https://github.com/qanastek/HugsVision/tree/main/recipes/pneumothorax/binary_classification/Image_Classifier.ipynb">
              <img src="https://raw.githubusercontent.com/qanastek/HugsVision/main/ressources/images/receipes/github_logo.png" height="32">
          </a>
      </td>
  </tr>
  <tr>
      <td align="center">
          <a href="https://colab.research.google.com/drive/1IIs3iWaVcH3sRkijdsXqQit0XXewJ0pJ?usp=sharing">
              <img src="https://raw.githubusercontent.com/qanastek/HugsVision/main/ressources/images/receipes/colab_logo.png" height="28">
          </a>
      </td>
  </tr>

  <!-- ------------------------------------------------------------------- -->
  
  <tr>
      <td rowspan="3" width="160">
        <img src="https://raw.githubusercontent.com/qanastek/HugsVision/main/ressources/images/receipes/new_blood_cells_coco.png" width="256">
      </td>    
      <td rowspan="3">
        <b>Training a End-To-End Object Detection with Transformers to detect blood cells:</b> A demonstration of how to train a E2E Object Detection Transformer model which can detect and identify blood cells with HugsVision.
      </td>
      <td align="center" width="80">
          <a href="https://nbviewer.jupyter.org/github/qanastek/HugsVision/blob/main/recipes/blood_cells/object_detection/Object_Detection.ipynb">
              <img src="https://raw.githubusercontent.com/qanastek/HugsVision/main/ressources/images/receipes/nbviewer_logo.svg" height="34">
          </a>
      </td>
  </tr>
  <tr>
      <td align="center">
          <a href="https://github.com/qanastek/HugsVision/tree/main/recipes/blood_cells/object_detection/Object_Detection.ipynb">
              <img src="https://raw.githubusercontent.com/qanastek/HugsVision/main/ressources/images/receipes/github_logo.png" height="32">
          </a>
      </td>
  </tr>
  <tr>
      <td align="center">
          <a href="https://colab.research.google.com/drive/1Q7_HYfZKrQJHV052OCGnZBHwKMIep3kv?usp=sharing">
              <img src="https://raw.githubusercontent.com/qanastek/HugsVision/main/ressources/images/receipes/colab_logo.png" height="28">
          </a>
      </td>
  </tr>

  <!-- ------------------------------------------------------------------- -->
  
  <tr>
      <td rowspan="3" width="160">
        <img src="https://raw.githubusercontent.com/qanastek/HugsVision/main/ressources/images/receipes/kvasir_v2.png" width="256">
      </td>    
      <td rowspan="3">  
        <b>Training a Transformer Image Classifier to help endoscopists:</b> A demonstration of how to train a Image Classifier Transformer model that can help endoscopists to automate detection of various anatomical landmarks, phatological findings or endoscopic procedures in the gastrointestinal tract with HugsVision.
      </td>
      <td align="center" width="80">
          <a href="https://nbviewer.jupyter.org/github/qanastek/HugsVision/blob/main/recipes/kvasir_v2/binary_classification/Kvasir_v2_Image_Classifier.ipynb">
              <img src="https://raw.githubusercontent.com/qanastek/HugsVision/main/ressources/images/receipes/nbviewer_logo.svg" height="34">
          </a>
      </td>
  </tr>
  <tr>
      <td align="center">
          <a href="https://github.com/qanastek/HugsVision/blob/main/recipes/kvasir_v2/binary_classification/Kvasir_v2_Image_Classifier.ipynb">
              <img src="https://raw.githubusercontent.com/qanastek/HugsVision/main/ressources/images/receipes/github_logo.png" height="32">
          </a>
      </td>
  </tr>
  <tr>
      <td align="center">
          <a href="https://colab.research.google.com/drive/1PMV-5c54ZlyoVh6dtkazaDdJR7I8VaqN?usp=sharing">
              <img src="https://raw.githubusercontent.com/qanastek/HugsVision/main/ressources/images/receipes/colab_logo.png" height="28">
          </a>
      </td>
  </tr>

  <!-- ------------------------------------------------------------------- -->
  
  <tr>
      <td rowspan="3" width="160">
        <img src="https://raw.githubusercontent.com/qanastek/HugsVision/main/ressources/images/receipes/HAM10000.png" width="256">
      </td>    
      <td rowspan="3">  
        <b>Training and using a TorchVision Image Classifier in 5 min to identify skin cancer:</b> A fast and easy tutorial to train a TorchVision Image Classifier that can help dermatologist in their identification procedures Melanoma cases with HugsVision and HAM10000 dataset.
      </td>
      <td align="center" width="80">
          <a href="https://nbviewer.jupyter.org/github/qanastek/HugsVision/blob/main/recipes/HAM10000/binary_classification/HAM10000_Image_Classifier.ipynb">
              <img src="https://raw.githubusercontent.com/qanastek/HugsVision/main/ressources/images/receipes/nbviewer_logo.svg" height="34">
          </a>
      </td>
  </tr>
  <tr>
      <td align="center">
          <a href="https://github.com/qanastek/HugsVision/blob/main/recipes/HAM10000/binary_classification/HAM10000_Image_Classifier.ipynb">
              <img src="https://raw.githubusercontent.com/qanastek/HugsVision/main/ressources/images/receipes/github_logo.png" height="32">
          </a>
      </td>
  </tr>
  <tr>
      <td align="center">
          <a href="https://colab.research.google.com/drive/1tfRpFTT1GJUgrcwHI0pYdAZ5_z0VSevJ?usp=sharing">
              <img src="https://raw.githubusercontent.com/qanastek/HugsVision/main/ressources/images/receipes/colab_logo.png" height="28">
          </a>
      </td>
  </tr>
</table>

# Model architectures

All the model checkpoints provided by 🤗 Transformers and compatible with our tasks can be seamlessly integrated from the huggingface.co model hub where they are uploaded directly by users and organizations.

Before starting implementing, please check if your model has an implementation in `PyTorch` by refering to [this table](https://huggingface.co/transformers/index.html#supported-frameworks).

🤗 Transformers currently provides the following architectures for Computer Vision:

1. **[ViT](https://huggingface.co/transformers/model_doc/vit.html)** (from Google Research, Brain Team) released with the paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/pdf/2010.11929.pdf), by Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby.
2. **[DeiT](https://huggingface.co/transformers/model_doc/deit.html)** (from Facebook AI and Sorbonne University) released with the paper [Training data-efficient image transformers & distillation through attention](https://arxiv.org/pdf/2012.12877.pdf) by Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, Hervé Jégou.
3. **[BEiT](https://huggingface.co/transformers/master/model_doc/beit.html)** (from Microsoft Research) released with the paper [BEIT: BERT Pre-Training of Image Transformers](https://arxiv.org/pdf/2106.08254.pdf) by Hangbo Bao, Li Dong and Furu Wei.
4. **[DETR](https://huggingface.co/transformers/model_doc/detr.html)** (from Facebook AI) released with the paper [End-to-End Object Detection with Transformers](https://arxiv.org/pdf/2005.12872.pdf) by Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov and Sergey Zagoruyko.

# Build PyPi package

Build: `python setup.py sdist bdist_wheel`

Upload: `twine upload dist/*`

# Citation

If you want to cite the tool you can use this:

```bibtex
@misc{HugsVision,
  title={HugsVision},
  author={Yanis Labrak},
  publisher={GitHub},
  journal={GitHub repository},
  howpublished={\url{https://github.com/qanastek/HugsVision}},
  year={2021}
}
```
