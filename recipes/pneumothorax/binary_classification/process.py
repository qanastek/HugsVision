import os
import os.path
from shutil import copyfile

from tqdm import tqdm

import pandas as pd

df = pd.read_csv("./train_data.csv")

img_in = "./small_train_data_set/small_train_data_set/"
img_out = "./data/"

for index, row in tqdm(df.iterrows())   :

    label = "pneumothorax" if row['target'] == 1 else "normal"

    path_in = img_in + row['file_name']
    path_out = img_out + label + "/" + row['file_name']

    # Check if the input image exist
    if not os.path.isfile(path_in):
        continue

    # Check if the output dir of the label exist
    if not os.path.isdir(img_out + label):
        os.mkdir(img_out + label)
        print("Directory for the label " + label + " created!")

    # Copy the image
    copyfile(path_in, path_out)
