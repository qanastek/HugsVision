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
  def __init__(self, model_path: str, transform=transformTorchVision, device=None):

    self.transform = transform
    
    if device == None:
      self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
      self.device = device

    self.model_path = model_path if model_path.endswith("/") else model_path + "/"
    
    self.model = torch.load(self.model_path + "best_model.pth", map_location=self.device)

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

    img = torch.unsqueeze(img, 0).to(self.device)

    # Predict and get the corresponding label identifier
    pred = self.model(img)

    # Get string label from index
    label = self.config["id2label"][str(torch.max(pred, 1)[1].item())]
    
    return label

  """
  🤔 Predict from one image path
  """
  def predict(self, img_path: str, save_preview=False):  
    return self.predict_image(img=Image.open(img_path), save_preview=save_preview)
