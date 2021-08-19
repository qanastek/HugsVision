import os
import random
from datetime import datetime

import torch
from transformers import DetrFeatureExtractor, DetrForObjectDetection, pipeline

import numpy as np

import cv2
from PIL import Image

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

class SemanticSegmentationInference:
    
  """
  🤗 Constructor for the semantic segmentation trainer
  """
  def __init__(self, feature_extractor, model):

    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    self.feature_extractor = feature_extractor
    self.model = model
    print("Model loaded!")
    
  """
  🎨 Apply the semantic segmentation mask on the base image
  """
  @staticmethod
  def applySegmentation(pil_img: Image.Image, outputs, threshold=0.9):

    # Generate 200 random colors for the visualization
    COLORS_RGB = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for i in range(200)]

    # Get probabilities for each mask
    probas = outputs.logits.softmax(-1)[0, :, :-1]
    # Apply a thresholding
    keep = probas.max(-1).values > threshold
    # Get the masks
    masks = outputs.pred_masks
        
    # Convert the pillow image to a cv2 one
    base_img = np.array(pil_img)[:, :, ::-1].copy()
    height, width = base_img.shape[:2]
    
    for i, p, mask, c in enumerate(zip(probas[keep], masks[0][keep], COLORS_RGB)):
    
      # Convert the mask array to a cv2 grayscale image
      mask_img = cv2.cvtColor(
          mask.detach().numpy(),
          cv2.COLOR_GRAY2BGR
      )
      # Resize at the same size as the base image
      mask_img = cv2.resize(mask_img, (width, height))[:, :, 0]

      # Apply the random color where the mask is
      base_img[mask_img > 0] = COLORS_RGB[i]

    return base_img

  """
  ⚙️ Predict the bounding boxes for each object in the image
  Return: image_with_masks
  """
  def predict(self, img_path: str, threshold=0.9):

    # Load the image
    image_array = Image.open(img_path)

    # Transform the image
    inputs = self.feature_extractor(images=image_array, return_tensors="pt")

    # Predict and get the masks
    outputs = self.model(**inputs)
    
    return SemanticSegmentationInference.applySegmentation(image_array, outputs, threshold)
