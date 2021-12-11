import os.path
import argparse
from PIL import Image

from hugsvision.inference.TorchVisionClassifierInference import TorchVisionClassifierInference

parser = argparse.ArgumentParser(description='Image classifier')
parser.add_argument('--path', type=str, default="./OUT_TORCHVISION/HAM10000/", help='The model path')
parser.add_argument('--img', type=str, default="/users/ylabrak/datasets/HAM10000/bcc/ISIC_0024331.jpg", help='The input image')
args = parser.parse_args()

classifier = TorchVisionClassifierInference(
    model_path = args.path,
    # device="cpu",
)

print("Process the image: " + args.img) 

label = classifier.predict(img_path=args.img)
print("Predicted class:", label)

label = classifier.predict_image(img=Image.open(args.img))
print("Predicted class:", label)
