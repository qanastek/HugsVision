import argparse
import os.path

from hugsvision.inference import VisionClassifierInference 

parser = argparse.ArgumentParser(description='Image classifier')
parser.add_argument('--path', type=str, default="./out/MYVITMODEL/1_2021-08-10-00-53-58/model/", help='The model path')
parser.add_argument('--img', type=str, default="1.jpg", help='The input image')
args = parser.parse_args() 

print("Process the image: " + args.img)

try:

    label = VisionClassifierInference(model_path: args.path).predict(img_path: './data/demo/' + args.img)
    print("Predicted class:", label)

except Exception as e:
    if "containing a preprocessor_config.json file" in str(e) and os.path.isfile(args.path + "config.json") == True:
        print("\033[91m\033[4mError:\033[0m")
        print("\033[91mRename the config.json file into \033[4mpreprocessor_config.json\033[0m")
    else:
        print(str(e))