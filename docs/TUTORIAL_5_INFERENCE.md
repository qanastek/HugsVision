# Tutorial 5: Inference

This part of the tutorial shows how you can use your trained models.

## TorchVisionClassifierInference Object

Firstly, you need to initiate the inference object:

```python
from transformers import DeiTFeatureExtractor, DeiTForImageClassification
from hugsvision.inference.VisionClassifierInference import VisionClassifierInference

classifier = VisionClassifierInference(
    feature_extractor = DeiTFeatureExtractor.from_pretrained("out/model/"),
    model = DeiTForImageClassification.from_pretrained("out/model/"),
)
```

Then, you can predict the images classes using one of the methods.

__Method 1 :__ From image path

```python
label = classifier.predict(img_path="samples/kvasir_v2/dyed-lifted-polyps.jpg")
print("Predicted class:", label)
```

__Method 2 :__ From Pillow image

```python
label = classifier.predict_image(img=Image.open("samples/kvasir_v2/dyed-lifted-polyps.jpg"))
print("Predicted class:", label)
```

## TorchVisionClassifierInference Object

Firstly, you need to initiate the inference object:

```python
from hugsvision.inference.TorchVisionClassifierInference import TorchVisionClassifierInference

classifier = TorchVisionClassifierInference(
    model_path = "out/model/",
    # device="cpu",
)
```

Then, you can predict the images classes using one of the methods.

__Method 1 :__ From image path

```python
label = classifier.predict(img_path="samples/kvasir_v2/dyed-lifted-polyps.jpg")
print("Predicted class:", label)
```

__Method 2 :__ From Pillow image

```python
label = classifier.predict_image(img=Image.open("samples/kvasir_v2/dyed-lifted-polyps.jpg"))
print("Predicted class:", label)
```

You can choose between: returning directly the class corresponding to the `argmax` output as a string (`return_str=True`) or returning a dictionary where you have the probability of each classes (`return_str=False`).

```python
label = classifier.predict(img_path="samples/kvasir_v2/dyed-lifted-polyps.jpg", return_str=False)
print("Vector of probabilities:", label)
```

## ObjectDetectionInference Object

Firstly, you need to initiate the inference object:

```python
from transformers import DetrFeatureExtractor, DetrForObjectDetection
from hugsvision.inference.ObjectDetectionInference import ObjectDetectionInference

inference = ObjectDetectionInference(
    DetrFeatureExtractor.from_pretrained("out/model/"),
    DetrForObjectDetection.from_pretrained("out/model/", from_tf=False)
)
```

Then, you can predict the bounding boxes of the input image using one of the two methods available.

__Method 1 :__ From image path

```python
image, probas, bboxes_scaled = inference.predict(
    "samples/blood_cells/42.jpg",
    threshold = 0.5,
    visualize=False,
)
```

__Method 2 :__ From Pillow Image

```python
image, probas, bboxes_scaled = inference.predict_img(
    Image.open("samples/blood_cells/42.jpg"),
    threshold = 0.5,
    visualize=False,
)
```

Both method output the bounding boxes coordinates and probabilities.

### Visualization

You can therefore applying bounding boxes to the original image :

```python
folder_name = str(int(time.time())) + "/"
Path(folder_name).mkdir(parents=True, exist_ok=True)
cpt = 0
for p, (xmin, ymin, xmax, ymax) in zip(probas, bboxes_scaled.tolist()):
    cropped_img = image.crop((xmin, ymin, xmax, ymax))
    cropped_img.save(folder_name + str(cpt) + ".png")
    cpt += 1
```

## Next documentation

You can now take a look at:
* [TUTORIAL_6_PUBLISH_HuggingFace](TUTORIAL_6_PUBLISH_HuggingFace.md)
* [TUTORIAL_7_SETUP_HuggingFace_Space](TUTORIAL_7_SETUP_HuggingFace_Space.md)
* [TUTORIAL_8_IMAGE_EMBEDDINGS](TUTORIAL_8_IMAGE_EMBEDDINGS.md)
