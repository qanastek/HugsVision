import argparse

from hugsvision.nnet.ObjectDetectionTrainer import ObjectDetectionTrainer

parser = argparse.ArgumentParser(description='Object Detection')
parser.add_argument('--name', type=str, default="MyDETRModel", help='The name of the model')
parser.add_argument('--train', type=str, default="./BCCD_COCO/train/", help='The directory of the train folder containing the _annotations.coco.json')
parser.add_argument('--dev', type=str, default="./BCCD_COCO/valid/", help='The directory of the dev folder containing the _annotations.coco.json')
parser.add_argument('--test', type=str, default="./BCCD_COCO/test/", help='The directory of the test folder containing the _annotations.coco.json')
parser.add_argument('--output', type=str, default="./out/", help='The output directory of the model')
parser.add_argument('--epochs', type=int, default=1, help='Number of Epochs')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
args = parser.parse_args() 

huggingface_model = "facebook/detr-resnet-50"
# huggingface_model = "facebook/detr-resnet-101"

# Train the model
trainer = ObjectDetectionTrainer(
	model_name = args.name,	
	output_dir = args.output,
	
	train_path = args.train,
	dev_path   = args.dev,
	test_path  = args.test,
	
	model_path = huggingface_model,

	max_epochs = args.epochs,
	batch_size = args.batch_size,
)

# Test on a single image
trainer.testing(img='../samples/blood_cells/42.jpg')