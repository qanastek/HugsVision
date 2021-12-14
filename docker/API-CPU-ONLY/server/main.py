# -*- coding: utf-8 -*-

from io import BytesIO
from typing import Optional, List

from fastapi import FastAPI, File, UploadFile

import numpy as np
from PIL import Image

from transformers import DeiTFeatureExtractor, DeiTForImageClassification
from hugsvision.inference.VisionClassifierInference import VisionClassifierInference
from hugsvision.inference.TorchVisionClassifierInference import TorchVisionClassifierInference

models = {}

models_name = open("models.txt","r").read().split("\n")

id2label = open("id2label.txt","r").read().split("\n")

app = FastAPI()

def predict_image(image, model_name):

    image = Image.fromarray(np.uint8(image)).convert('RGB')

    model_path = "./models/" + model_name

    if model_name == "DeiT" and model_name not in models:

        models[model_name] = VisionClassifierInference(
            feature_extractor = DeiTFeatureExtractor.from_pretrained(model_path),
            model = DeiTForImageClassification.from_pretrained(model_path),
        )

    elif model_name not in models:

        models[model_name] = TorchVisionClassifierInference(
            model_path = model_path
        )

    pred = models[model_name].predict_image(img=image, return_str=False)

    for key in pred.keys():
        pred[key] = pred[key]/100
    
    idx = list(pred.values()).index(max(list(pred.values())))
    label = list(pred.keys())[idx]

    return {
        "label": label,
        "probabilities": pred,
    }

@app.get("/")
def read_root():
    return { "documentation": "http://127.0.0.1:8000/docs#/" }

@app.get("/models")
def read_models():
    return { "models": models_name }

@app.post("/predict/{model_name}")
async def read_item(model_name: str = models_name[-1], file: UploadFile = File(...)):
    image = Image.open(BytesIO(await file.read()))
    return predict_image(image,model_name)

@app.post("/predicts/{model_name}")
async def read_item(model_name: str = models_name[-1], files: List[UploadFile] = File(...)):
    res = []
    for file in files:
        image = Image.open(BytesIO(await file.read()))
        res.append(predict_image(image,model_name))
    return res
