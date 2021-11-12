import json

import torch
from torchvision import transforms

from PIL import Image

class TorchVisionClassifierInference:

  transformTorchVision = transforms.Compose([
      transforms.Resize((224,224), interpolation=Image.NEAREST),
      transforms.ToTensor(),
  ])

  """
  🤗 Constructor for the image classifier trainer of TorchVision
  """
  def __init__(self, model_path: str, transform=transformTorchVision):

    self.model_path = model_path if model_path.endswith("/") else model_path + "/"
    self.transform = transform
    
    self.model = torch.load(self.model_path + "best_model.pth", map_location="cuda:0")

    self.config = json.load(open(self.model_path + "config.json", "r"))

    print("Model loaded!")

  """
  🤔 Predict from one image at the Pillow format
  """
  def predict_image(self, img, save_preview=False):
    
    img = self.transform(img)

    if save_preview:
      pil_img = transforms.ToPILImage()(img)
      pil_img.save("preview.jpg")

    img = torch.unsqueeze(img, 0)

    # Predict and get the corresponding label identifier
    pred = self.model(img)

    # Get label index
    _, predicted_class_idx = torch.max(pred, 1)

    # Get string label from index
    label = self.config["id2label"][str(predicted_class_idx.item())]
    
    return label

  """
  🤔 Predict from one image path
  """
  def predict(self, img_path: str):  
    return self.predict_image(img=Image.open(img_path))
