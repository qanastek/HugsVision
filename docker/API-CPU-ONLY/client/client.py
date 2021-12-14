import requests

from tqdm import tqdm

def predict(model_id, image_path):
    return requests.post(
        f"http://127.0.0.1:8000/predict/{model_id}",
        files = {
            'file': open(image_path, 'rb')
        }
    ).json()

def predict_all(model_id, image_paths):
    return requests.post(
        f"http://127.0.0.1:8000/predicts/{model_id}",
        files = [('files', open(path, 'rb')) for path in image_paths]
    ).json()

print(predict_all("ShuffleNetV2", ["images/akiec.jpg", "images/bcc.jpg"]))

for i in tqdm(range(1000)):
    res = predict("ShuffleNetV2","images/akiec.jpg")
