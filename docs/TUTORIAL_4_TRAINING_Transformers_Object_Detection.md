# Tutorial 4: Training a Transformers Object Detection model from HuggingFace

This part of the tutorial shows how you can train and save a transformers model from the HuggingFace platform.

## ObjectDetectionTrainer Object

### Prepare dataset

First, we need to download the dataset, in our case it's [BCCD (COCO)](https://www.kaggle.com/ammarnassanalhajali/bccd-coco) and weight around ~9 MB.

Once its as been downloaded, rename the annotation file like `_annotations.coco.json` for all the subsets:

```bash
mv anns.json _annotations.coco.json
```

You should have this structure:

```plain
BCCD_COCO
├── train
│   ├── _annotations.coco.json
│   ├── BloodImage_1.jpg
│   ├── BloodImage_2.jpg
│   └── BloodImage_3.jpg
├── valid
│   ├── _annotations.coco.json
│   ├── BloodImage_4.jpg
│   ├── BloodImage_5.jpg
│   └── BloodImage_6.jpg
├── test
│   ├── _annotations.coco.json
│   ├── BloodImage_7.jpg
│   ├── BloodImage_8.jpg
│   └── BloodImage_9.jpg
```

### Choose a object detection model on HuggingFace

Now we can choose our base model on which we will perform a fine-tuning to make it fit our needs.

Our choices aren't very large since they are only two models available yet on HuggingFace for this task and both are DETR based.

So, to be sure that the model will be compatible with `HugsVision` we need to have a model exported in `PyTorch` and compatible with the `object-detection` task obviously.

Models available with this criterias: https://huggingface.co/models?filter=pytorch&pipeline_tag=object-detection&sort=downloads

At the time I'am writing this, I recommand to use the following models:

* `facebook/detr-resnet-50`
* `facebook/detr-resnet-101`

Our model:

```python
huggingface_model = 'facebook/detr-resnet-50'
```

### Train the model

So, once the model choosen, we can start building the `Trainer` and start the fine-tuning :

```python
from hugsvision.nnet.ObjectDetectionTrainer import ObjectDetectionTrainer

trainer = ObjectDetectionTrainer(
    model_name = "MyDETRModel",
    output_dir = "./out/",
    train_path = "./BCCD_COCO/train/",
    dev_path   = "./BCCD_COCO/dev/",
    test_path  = "./BCCD_COCO/test/",
    model_path = huggingface_model,
    max_epochs = 1,
    batch_size = 4
)
```

## Resources

The best way of understanding how its works is to refer to the different recipes available in the `recipes` directory.

## Next documentation

You can now take a look at:
* [TUTORIAL_4_INFERENCE](TUTORIAL_4_INFERENCE.md)
