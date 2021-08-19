import argparse

from hugsvision.nnet.SemanticSegmentationTrainer import SemanticSegmentationTrainer

parser = argparse.ArgumentParser(description='Object Detection')
parser.add_argument('--name', type=str, default="MyDETRModel", help='The name of the model')
parser.add_argument('--output', type=str, default="./out/", help='The output directory of the model')
parser.add_argument('--epochs', type=int, default=1, help='Number of Epochs')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
args = parser.parse_args() 

huggingface_model = "facebook/detr-resnet-50-panoptic"

# Train the model
trainer = SemanticSegmentationTrainer(
	model_name = args.name,	
	output_dir = args.output,
	
	img_folder = '/users/ylabrak/datasets/coco_2017/val2017',
	ann_folder = '/users/ylabrak/datasets/coco_2017/panoptic_annotations_trainval2017/annotations/panoptic_val2017/panoptic_val2017/',
	ann_file   = '/users/ylabrak/datasets/coco_2017/panoptic_annotations_trainval2017/annotations/panoptic_val2017.json',
	
	model_path = huggingface_model,

	max_epochs = args.epochs,
	batch_size = args.batch_size,
)

# Test on a single image
trainer.testing(img_path='../../../samples/blood_cells/42.jpg')