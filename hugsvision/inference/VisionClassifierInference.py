import torch
from PIL import Image

from transformers import ViTFeatureExtractor, ViTForImageClassification, pipeline

class VisionClassifierInference:
    
  """
  🤗 Constructor for the image classifier trainer
  """
  def __init__(self, feature_extractor, model, resolution=128):
    
    self.feature_extractor = feature_extractor
    self.model = model
    print("Model loaded!")
    
    self.resolution = resolution

  """
  Arguments
  ---------
  img: Pillow Image
  """
  def predict_image(self, img, return_str=True):

    # Change resolution to 128x128
    img.thumbnail((self.resolution,self.resolution))

    # Transform the image
    encoding = self.feature_extractor(images=img, return_tensors="pt")

    # Predict and get the corresponding label identifier
    pred = self.model(encoding['pixel_values'])

    # Get label index
    predicted_class_idx = pred.logits.argmax(-1).tolist()[0]

    if return_str:
      
      # Get string label from index
      return self.model.config.id2label[predicted_class_idx]

    else:

      labels = list(self.model.config.label2id.keys())
      
      softmax = torch.nn.Softmax(dim=0)
      
      pred_soft = softmax(pred[0][0])
      pred_soft = torch.mul(pred_soft, 100)
      probabilities = pred_soft.tolist()

      return dict(zip(labels, probabilities))

  """
  Arguments
  ---------
  img_path: str
  """
  def predict(self, img_path: str, return_str=True):  
    return self.predict_image(img=Image.open(img_path), return_str=return_str)
