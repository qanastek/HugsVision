import argparse
import os.path

from transformers import DetrFeatureExtractor, DetrForObjectDetection
from hugsvision.inference.ObjectDetectionInference import ObjectDetectionInference

parser = argparse.ArgumentParser(description='Image classifier')
parser.add_argument('--path', type=str, default="/users/ylabrak/Visual Transformers - ViT2/server_DETR/out/model/2021-08-15-03-24-46/", help='The model path')
parser.add_argument('--img', type=str, default="../samples/blood_cells/42.jpg", help='The input image')
args = parser.parse_args() 

print("Process the image: " + args.img)

inference = ObjectDetectionInference(
    DetrFeatureExtractor.from_pretrained(args.path),
    DetrForObjectDetection.from_pretrained(args.path, from_tf=False)
)

inference.predict(args.img)
