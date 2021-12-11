# Tutorial 3: Training a TorchVision model

This part of the tutorial shows how you can train and save a TorchVision model.

It will give you a quick intro to training TorchVision models such as VGG16, Inception V3, Resnet50, AlexNet and much more with HugsVision.

## TorchVisionClassifierTrainer Object

The `VisionClassifierTrainer` class contains the functions needed to train the TorchVision model taken from the [TorchVision website](https://pytorch.org/vision/stable/models.html).

**Note:** The previous steps for loading the dataset with the `torch_vision=True` argument need to be already done.

### Choose a image classifier model on TorchVision

Now we can choose our base model on which we will perform a fine-tuning to make it fit our needs.

So, to be sure that the model will be compatible with `HugsVision` we need to have a model exported in `PyTorch` and listed in the [TORCHVISION.MODELS](https://pytorch.org/vision/stable/models.html) section of the `PyTorch` documentation.

You can also find the list of available models by typing directly in Python:

```python
import torchvision.models as models
print(models.__dict__.keys())
```

Output:

```python
dict_keys(['alexnet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19', 'squeezenet1_0', 'squeezenet1_1', 'inception_v3', 'densenet121', 'densenet169', 'densenet201', 'densenet161', 'googlenet', 'mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small', 'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0'])
```

Others models such as ```shufflenet_v2_x1_5 shufflenet_v2_x2_0``` wasn't implemented yet by the PyTorch team and return a ```NotImplementedError```.

### Dataloader

So, once the model chosen, we need to load the dataset using the `VisionDataset` class.

**Note**: Some rare architectures need to apply a transformation function to your training data to fit the model requirements. Globally, most of the models are expecting `224x224` resolutions but there's some counter examples in which you will need to add:

```python
torch_vision=True,
transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])
```

### Train the model

So, once the model chosen, we can start building the `Trainer` and start the fine-tuning.

```python
from hugsvision.nnet.TorchVisionClassifierTrainer import TorchVisionClassifierTrainer

trainer = TorchVisionClassifierTrainer(
    output_dir   = "./out_torchvision/HAM10000/",
    model_name   = 'densenet121',
    train        = train,
    test         = test,
    batch_size   = 64,
    max_epochs   = 100,
    id2label     = id2label,
    label2id     = label2id,
    lr=1e-3,
)
```

### Evaluate F1-Score

Using the F1-Score metrics will allow us to get a better representation of predictions for all the labels and find out if their are any anomalies wit ha specific label.

By default, the trainer is running a evaluation after each epoch with the latest model and at the end of training phase with the best model but you can also run it manually if you want to collect the list of predictions after the `argmax` function.

```python
hyp, ref = trainer.evaluate_f1_score()
```

Output:

```plain
              precision    recall  f1-score   support

       akiec       0.80      0.80      0.80        10
         bcc       0.80      0.89      0.84         9
         bkl       0.80      0.80      0.80        10
          df       1.00      1.00      1.00        15
         mel       0.75      0.67      0.71         9
          nv       0.92      0.86      0.89        14
        vasc       0.93      1.00      0.97        14

    accuracy                           0.88        81
   macro avg       0.86      0.86      0.86        81
weighted avg       0.88      0.88      0.88        81
```

## Resources

The best way of understanding how its works is to refer to the different recipes available in the `recipes` directory.

## Next documentation

You can now take a look at:
* [TUTORIAL_4_INFERENCE](TUTORIAL_4_INFERENCE.md)
