import argparse
import torchvision.models as models

from hugsvision.dataio.VisionDataset import VisionDataset
from hugsvision.nnet.TorchVisionClassifierTrainer import TorchVisionClassifierTrainer

parser = argparse.ArgumentParser(description='Image classifier')
parser.add_argument('--imgs', type=str, default="/users/ylabrak/datasets/HAM10000/", help='The directory of the input images')
parser.add_argument('--epochs', type=int, default=1, help='Number of Epochs')
parser.add_argument('--output', type=str, default="./OUT_TORCHVISION/HAM10000-wide_resnet50_2/", help='The output directory of the model')
args = parser.parse_args()

# Load the dataset
train, test, id2label, label2id = VisionDataset.fromImageFolder(
	args.imgs,
	test_ratio=0.10,
	balanced=True,
	torch_vision=True,
)

# Train the model
trainer = TorchVisionClassifierTrainer(
	output_dir   = args.output,
	model_name   = "wide_resnet50_2",
	# model_name   = "shufflenet_v2_x0_5",
	# model_name   = "mnasnet0_5",
	# model_name   = "resnext50_32x4d",
	# model_name   = "googlenet",
	# model_name   = "mobilenet_v3_small",
	# model_name   = "mobilenet_v3_large",
	# model_name   = "mobilenet_v2",
	# model_name   = "alexnet",
	# model_name   = "resnet18",
	# model_name   = "resnet50",
	# model_name   = "vgg16",
	train      	 = train,
	test      	 = test,
	batch_size   = 32,
	max_epochs   = args.epochs,
	id2label 	 = id2label,
	label2id 	 = label2id,
)

# Evaluate on the test sub-dataset
ref, hyp = trainer.evaluate_f1_score()