import argparse
import os.path

from transformers import ViTFeatureExtractor, ViTForImageClassification, pipeline
from PIL import Image

parser = argparse.ArgumentParser(description='Image classifier')
parser.add_argument('--path', type=str, default="./out/MYVITMODEL/1_2021-08-10-00-53-58/model/", help='The model path')
parser.add_argument('--img', type=str, default="1.jpg", help='The input image')
args = parser.parse_args() 

print("Process the image: " + args.img)

try:

    print("Load model from: " + args.path)
    feature_extractor = ViTFeatureExtractor.from_pretrained(args.path)
    model = ViTForImageClassification.from_pretrained(args.path)

    # Load the image
    image_array = Image.open('./data/demo/' + args.img)

    # Change resolution to 128x128
    image_array.thumbnail((128,128))

    # Transform the image
    encoding = feature_extractor(images=image_array, return_tensors="pt")

    # Predict and get the corresponding label identifier
    pred = model(encoding['pixel_values'])

    predicted_class_idx = pred.logits.argmax(-1).tolist()[0]

    print(pred.logits)
    print("Predicted class:", model.config.id2label[predicted_class_idx], "(", str(predicted_class_idx), ")")

except Exception as e:
    if "containing a preprocessor_config.json file" in str(e) and os.path.isfile(args.path + "config.json") == True:
        print("\033[91m\033[4mError:\033[0m")
        print("\033[91mRename the config.json file into \033[4mpreprocessor_config.json\033[0m")
    else:
        print(str(e))