import argparse

from hugsvision.nnet.VisionClassifierTrainer import VisionClassifierTrainer
from hugsvision.dataio.VisionDataset import VisionDataset

from torchvision.datasets import ImageFolder
from transformers import ViTFeatureExtractor, ViTForImageClassification

parser = argparse.ArgumentParser(description='Image classifier')
parser.add_argument('--name', type=str, default="MyVitModel", help='The name of the model')
parser.add_argument('--imgs', type=str, default="./images/", help='The directory of the input images')
parser.add_argument('--output', type=str, default="./out/", help='The output directory of the model')
parser.add_argument('--metric', type=str, default="f1", help='The metric')
parser.add_argument('--epochs', type=int, default=1, help='Number of Epochs')
args = parser.parse_args() 

# Load the dataset
myDataset = ImageFolder(args.imgs)

# Both way indexes
num_labels, label2id, id2label = VisionDataset.getConfig(myDataset)

huggingface_model = 'google/vit-base-patch16-224-in21k'

# Train the model
trainer = VisionClassifierTrainer(
	model_name = args.name,
	dataset = myDataset,
	output_dir = args.output,
	max_epochs = args.epochs,
	test_ratio = 0.15,
	dev_ratio = 0.15,
    lr = 2e-5,
	batch_size = 32,
	cores = 4,
	fp16=True,
	balanced=True,
	augmentation=True,
	eval_metric = args.metric,
	ids2labels = id2label,
	model = ViTForImageClassification.from_pretrained(
	    huggingface_model,
	    num_labels = num_labels,
	    label2id = label2id,
	    id2label = id2label
	),
	feature_extractor = ViTFeatureExtractor.from_pretrained(
		huggingface_model,
	),
)

# Evaluate on the test sub-dataset
hyp, ref = trainer.evaluate_f1_score()

# # Test on a single image
# trainer.testing(img='./data/demo/42.png',expected=2)
# trainer.testing(img='./data/demo/3.jpg',expected=0)
# trainer.testing(img='./data/demo/5.jpg',expected=2)
# trainer.testing(img='./data/demo/4.jpg',expected=1)