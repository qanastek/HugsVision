import os

import cv2
import numpy as np
from tqdm import tqdm

class ComputeImgMeanStd:

  """
  🎨 Compute mean and std for the normalization phase
  """
  def compute(dataset_root :str, image_size=224):

    print("Compute mean and std for the normalization...")

    dataset_root = dataset_root if dataset_root.endswith("/") else dataset_root + "/"
    
    # Get images paths
    image_paths = []
    for label in os.listdir(dataset_root):
        for image_name in os.listdir(dataset_root + label):
            image_paths.append(dataset_root + label + "/" + image_name)

    imgs = []
    means, stds = [], []

    # Load images as OpenCV2 images
    for i in tqdm(range(len(image_paths))):
        img = cv2.imread(image_paths[i])
        img = cv2.resize(img, (image_size, image_size))
        imgs.append(img)

    # Stack by colors
    imgs = np.stack(imgs, axis=3)
    print(imgs.shape)

    # Put on 0-255 range
    imgs = imgs.astype(np.float32) / 255.

    # For RGB
    for i in range(3):
        pixels = imgs[:, :, i, :].ravel()
        means.append(np.mean(pixels))
        stds.append(np.std(pixels))

    # Reorder colors
    means.reverse()
    stds.reverse()

    print("normMean = {}".format(means))
    print("normStd = {}".format(stds))
    
    return means, stds