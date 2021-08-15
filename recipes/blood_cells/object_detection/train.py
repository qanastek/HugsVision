import argparse

from hugsvision.nnet.ObjectDetectionTrainer import ObjectDetectionTrainer

parser = argparse.ArgumentParser(description='Object Detection')
parser.add_argument('--name', type=str, default="MyDETRModel", help='The name of the model')
parser.add_argument('--train', type=str, default="./bccd_coco/train/", help='The directory of the train folder containing the _annotations.coco.json')
parser.add_argument('--dev', type=str, default="./bccd_coco/valid/", help='The directory of the dev folder containing the _annotations.coco.json')
parser.add_argument('--test', type=str, default="./bccd_coco/test/", help='The directory of the test folder containing the _annotations.coco.json')
parser.add_argument('--output', type=str, default="./out/", help='The output directory of the model')
args = parser.parse_args() 

PATH = "/users/ylabrak/Visual Transformers - ViT2/server_DETR/BCCD (COCO)"

huggingface_model = "facebook/detr-resnet-50"
# huggingface_model = "facebook/detr-resnet-101"

# Train the model
trainer = ObjectDetectionTrainer(
	model_name = args.name,	
	train_path = PATH + "/train/",
	dev_path   = PATH + "/test/",
	test_path  = PATH + "/valid/",
	output_dir = args.output,
	max_epochs = 1,
	batch_size = 4,
	model_path = huggingface_model,
)

# Test on a single image
trainer.testing(img='../samples/blood_cells/42.jpg')