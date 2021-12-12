import gradio as gr

import numpy as np
from PIL import Image

from transformers import DeiTFeatureExtractor, DeiTForImageClassification
from hugsvision.inference.VisionClassifierInference import VisionClassifierInference
from hugsvision.inference.TorchVisionClassifierInference import TorchVisionClassifierInference

models_name = [
    "VGG16",
    "DeiT",
    "DenseNet121",
    "MobileNetV2",
    "ShuffleNetV2",
]

radio = gr.inputs.Radio(models_name, default="DenseNet121", type="value")

def predict_image(image, model_name):

    image = Image.fromarray(np.uint8(image)).convert('RGB')

    model_path = "./models/" + model_name

    if model_name == "DeiT":

        model = VisionClassifierInference(
            feature_extractor = DeiTFeatureExtractor.from_pretrained(model_path),
            model = DeiTForImageClassification.from_pretrained(model_path),
        )

    else:

        model = TorchVisionClassifierInference(
            model_path = model_path
        )

    pred = model.predict_image(img=image, return_str=False)

    for key in pred.keys():
        pred[key] = pred[key]/100
    
    return pred

id2label = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

samples = [["images/" + p + ".jpg"] for p in id2label]
print(samples)

image = gr.inputs.Image(shape=(224, 224), label="Upload Your Image Here")
label = gr.outputs.Label(num_top_classes=len(id2label))

interface = gr.Interface(
    fn=predict_image, 
    inputs=[image,radio], 
    outputs=label, 
    capture_session=True, 
    allow_flagging=False,
    thumbnail="ressources/thumbnail.png",
    article="""
<html style="color: white;">
<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-v0zy{background-color:#efefef;color:#000000;font-weight:bold;text-align:center;vertical-align:top}
.tg .tg-4jb6{background-color:#ffffff;color:#333333;text-align:center;vertical-align:top}
</style>
<table class="tg">
<thead>
<tr>
<th class="tg-v0zy">Model</th>
<th class="tg-v0zy">Accuracy</th>
<th class="tg-v0zy">Size</th>
</tr>
</thead>
<tbody>
<tr>
<td class="tg-4jb6">VGG16</td>
<td class="tg-4jb6">38.27%</td>
<td class="tg-4jb6">512.0 MB</td>
</tr>
<tr>
<td class="tg-4jb6">DeiT</td>
<td class="tg-4jb6">71.60%</td>
<td class="tg-4jb6">327.0 MB</td>
</tr>
<tr>
<td class="tg-4jb6">DenseNet121</td>
<td class="tg-4jb6">77.78%</td>
<td class="tg-4jb6">27.1 MB</td>
</tr>
<tr>
<td class="tg-4jb6">MobileNetV2</td>
<td class="tg-4jb6">75.31%</td>
<td class="tg-4jb6">8.77 MB</td>
</tr>
<tr>
<td class="tg-4jb6">ShuffleNetV2</td>
<td class="tg-4jb6">76.54%</td>
<td class="tg-4jb6">4.99 MB</td>
</tr>
</tbody>
</table>
</html>
    """,
    theme="darkhuggingface",
    title="HAM10000: Training and using a TorchVision Image Classifier in 5 min to identify skin cancer",
    description="A fast and easy tutorial to train a TorchVision Image Classifier that can help dermatologist in their identification procedures Melanoma cases with HugsVision and HAM10000 dataset.",
    allow_screenshot=True,
    show_tips=False,
    encrypt=False,
    examples=samples,
)
interface.launch()