# Labels (diagnostic categories): https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T

import os
import shutil
import pandas as pd
from tqdm import tqdm

metadata = pd.read_csv("HAM10000_metadata.csv").set_index('image_id').T.to_dict('list')

for current_dir in ["HAM10000_images_part_1","HAM10000_images_part_2"]:

    for image_path in tqdm(os.listdir(current_dir)):

        label = metadata[image_path.split(".")[0]][1]
        os.makedirs("./HAM10000/" + label, exist_ok=True)
        shutil.copy2(current_dir + "/" + image_path, "./HAM10000/" + label + "/" + image_path)
