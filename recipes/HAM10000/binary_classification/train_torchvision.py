import argparse
from datetime import datetime

from hugsvision.dataio.VisionDataset import VisionDataset
from hugsvision.nnet.TorchVisionClassifierTrainer import TorchVisionClassifierTrainer

parser = argparse.ArgumentParser(description='Image classifier')
parser.add_argument('--imgs', type=str, default="/users/ylabrak/datasets/HAM10000/", help='The directory of the input images')
parser.add_argument('--epochs', type=int, default=100, help='Number of Epochs')
parser.add_argument('--output', type=str, default="./out_torchvision/HAM10000-best/", help='The output directory of the model')
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
	output_dir   = args.output + str(datetime.today().strftime("%Y-%m-%d-%H-%M-%S")) + "/",
	model_name   = "densenet121",
	train      	 = train,
	test      	 = test,
	batch_size   = 64,
	max_epochs   = args.epochs,
	id2label 	 = id2label,
	label2id 	 = label2id,
	lr=1e-3,
)
