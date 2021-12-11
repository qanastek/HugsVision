# Tutorial 1: Loading Training Data

This part of the tutorial shows how you can load a corpus for training any models.

## VisionDataset Object

The `VisionDataset` class contains the functions needed to generate the corresponding subset of images that you will use to `train` and `test` the model.

Import:

```python
from hugsvision.dataio.VisionDataset import VisionDataset
```

### Generate a dataset

1. The first one consist of calling the `VisionDataset.fromImageFolder` method in which you will pass a single argument of type string, called `dataset` that refer to the root path of the dataset directory in which each sub-directory refer to a class.

    Directory structure:

    ```plain
    dataset
    ├── cat
    │   ├── cat1.jpg
    │   └── cat2.jpg
    ├── dog
    │   ├── dog1.jpg
    │   ├── dog2.jpg
    │   └── dog3.jpg
    ```

    Snippet:

    ```python
    train, test, id2label, label2id = VisionDataset.fromImageFolder(
        "./users/MyUser/datasets/animals/",
        test_ratio   = 0.15,
        balanced     = True,
    )
    ```

2. The second one consist of calling the `VisionDataset.fromImageFolders` method in which you will pass two separate arguments `train` and `test` that refer to the subset root directories paths. The processing of each subset is similar to the first method.

    Directories structures:

    ```plain
    train
    ├── cat
    │   ├── cat1.jpg
    │   └── cat2.jpg
    ├── dog
    │   ├── dog1.jpg
    │   ├── dog2.jpg
    │   └── dog3.jpg
    test
    ├── cat
    │   ├── cat3.jpg
    │   └── cat4.jpg
    ├── dog
    │   ├── dog4.jpg
    │   └── dog5.jpg
    ```

    Snippet:

    ```python
    train, test, id2label, label2id = VisionDataset.fromImageFolder(
        "./users/MyUser/datasets/animals/train/",
        "./users/MyUser/datasets/animals/test/",
        balanced     = True,
    )
    ```

### Compatibilities with TorchVision

If you want to train a _TorchVision_ models with `TorchVisionClassifierTrainer`, you will need to load the data differently by simply adding the `torch_vision=True` argument to the dataset loader like this :

```python
train, test, id2label, label2id = VisionDataset.fromImageFolder(
    "./users/MyUser/datasets/animals/",
    balanced     = True,
    torch_vision = True,
)
```

**Note**: Some rare architectures need to apply a transformation function to your training data to fit the model requirements. Globally, most of the models are expecting `224x224` resolutions but there's some counter examples in which you will need to add the following parameter tweaked with the model requirements :

```python
train, test, id2label, label2id = VisionDataset.fromImageFolder(
    "./users/MyUser/datasets/animals/",
    balanced     = True,
    torch_vision = True,
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ]),
)
```

### Labels dictionaries (id2label & label2id)

During the data loading phase, both dictionaries id2label and label2id are generated.

```json
{
    "num_classes": 2,
    "hidden_size": 1024,
    "id2label": {
        "0": "cat",
        "1": "dog",
    },
    "label2id": {
        "cat": "0",
        "dog": "1",
    },
    "architectures": [
        "densenet121"
    ]
}
```

They are then saved during the training phase in the `config.json` file available in the model directory and can be publicly accessed at `TorchVisionClassifierTrainer.config` or `TorchVisionClassifierInference.config`.

## Resources

The best way of understanding how its works is to refer to the different recipes available in the `recipes` directory.

## Next documentation

You can now take a look at:
* [TUTORIAL_2_TRAINING_Transformers](TUTORIAL_2_TRAINING_Transformers.md)
* [TUTORIAL_3_TRAINING_TorchVision](TUTORIAL_3_TRAINING_TorchVision.md)
