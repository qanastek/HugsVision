import argparse

from hugsvision.dataio.VisionDataset import VisionDataset

from hugsvision.nnet.VisionClassifierTrainer import VisionClassifierTrainer
from transformers import DeiTFeatureExtractor, DeiTForImageClassification

parser = argparse.ArgumentParser(description='Image classifier')
parser.add_argument('--name', type=str, default="KVASIR_V2", help='The name of the model')
parser.add_argument('--imgs', type=str, default="/users/ylabrak/datasets/kvasir-dataset-v2/", help='The directory of the input images')
parser.add_argument('--output', type=str, default="./out/", help='The output directory of the model')
parser.add_argument('--epochs', type=int, default=1, help='Number of Epochs')
args = parser.parse_args() 

# Load the dataset
train, test, id2label, label2id = VisionDataset.fromImageFolder(
	args.imgs,
	test_ratio=0.15,
	balanced=True,
	augmentation=True,
)

huggingface_model = "facebook/deit-base-distilled-patch16-224"

# Train the model
trainer = VisionClassifierTrainer(
	model_name   = args.name,
	train      	 = train,
	test      	 = test,
	output_dir   = args.output,
	max_epochs   = args.epochs,
	batch_size   = 32, # On RTX 2080 Ti
	test_ratio   = 0.15,
    lr 		     = 2e-5,
	fp16	     = True,
	balanced     = True,
	augmentation = True,
	model = DeiTForImageClassification.from_pretrained(
	    huggingface_model,
	    num_labels = len(label2id),
	    label2id   = label2id,
	    id2label   = id2label
	),
	feature_extractor = DeiTFeatureExtractor.from_pretrained(
		huggingface_model,
	),
)

# Evaluate on the test sub-dataset
hyp, ref = trainer.evaluate_f1_score()