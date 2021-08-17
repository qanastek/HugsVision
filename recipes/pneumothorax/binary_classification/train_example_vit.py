import argparse

from hugsvision.dataio.VisionDataset import VisionDataset
from hugsvision.nnet.VisionClassifierTrainer import VisionClassifierTrainer

from transformers import ViTFeatureExtractor, ViTForImageClassification

parser = argparse.ArgumentParser(description='Image classifier')
parser.add_argument('--name', type=str, default="MyVitModel", help='The name of the model')
parser.add_argument('--imgs', type=str, default="./images/", help='The directory of the input images')
parser.add_argument('--output', type=str, default="./out/", help='The output directory of the model')
parser.add_argument('--epochs', type=int, default=1, help='Number of Epochs')
args = parser.parse_args() 

# Load the dataset
train, test, id2label, label2id = VisionDataset.fromImageFolder(
	args.imgs,
	test_ratio   = 0.15,
	balanced     = True,
	augmentation = True,
)

# # Load the dataset
# train, test, id2label, label2id = VisionDataset.fromImageFolders(
# 	"/<PATH>/train/",
# 	"/<PATH>/test/",
# )

huggingface_model = 'google/vit-base-patch16-224-in21k'

# Train the model
trainer = VisionClassifierTrainer(
	model_name   = args.name,
	train      	 = train,
	test      	 = test,
	output_dir   = args.output,
	max_epochs   = args.epochs,
	cores 	     = 4,
	batch_size   = 32,
	test_ratio   = 0.15,
	balanced     = True,
	augmentation = True,
	model = ViTForImageClassification.from_pretrained(
	    huggingface_model,
	    num_labels = len(label2id),
	    label2id   = label2id,
	    id2label   = id2label
	),
	feature_extractor = ViTFeatureExtractor.from_pretrained(
		huggingface_model,
	),
)

# Evaluate on the test sub-dataset
# print(trainer.evaluate())
hyp, ref = trainer.evaluate_f1_score()

# # Test on a single image
# trainer.testing(img='./data/demo/42.png',expected=2)
# trainer.testing(img='./data/demo/3.jpg',expected=0)
# trainer.testing(img='./data/demo/5.jpg',expected=2)
# trainer.testing(img='./data/demo/4.jpg',expected=1)