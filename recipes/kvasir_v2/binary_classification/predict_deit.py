import argparse
import os.path

from transformers import DeiTFeatureExtractor, DeiTForImageClassification

from hugsvision.inference.VisionClassifierInference import VisionClassifierInference

parser = argparse.ArgumentParser(description='Image classifier')
parser.add_argument('--path', type=str, default="./out/KVASIR_V2_MODEL_DEIT/20_2021-08-20-01-46-44/model/", help='The model path')
parser.add_argument('--img', type=str, default="../../../samples/kvasir_v2/dyed-lifted-polyps.jpg", help='The input image')
args = parser.parse_args() 

print("Process the image: " + args.img)

try:

    classifier = VisionClassifierInference(
        feature_extractor = DeiTFeatureExtractor.from_pretrained(args.path),
        model = DeiTForImageClassification.from_pretrained(args.path),
    )

    label = classifier.predict(img_path=args.img)

    print("Predicted class:", label)

except Exception as e:
    if "containing a preprocessor_config.json file" in str(e) and os.path.isfile(args.path + "config.json") == True:
        print("\033[91m\033[4mError:\033[0m")
        print("\033[91mRename the config.json file into \033[4mpreprocessor_config.json\033[0m")
    else:
        print(str(e))