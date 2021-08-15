import os
import torchvision

"""
🖼 COCO dataset at the DETR format
"""
class CocoDetection(torchvision.datasets.CocoDetection):

    def __init__(self, img_folder, feature_extractor):

        # Get path image annotations
        ann_file = os.path.join(img_folder, "_annotations.coco.json")
        
        # Default constructor
        super(CocoDetection, self).__init__(img_folder, ann_file)
        
        # Load the feature extractor
        self.feature_extractor = feature_extractor

    def __getitem__(self, idx):

        # Read in PIL image and target in COCO format
        img, target = super(CocoDetection, self).__getitem__(idx)
        
        # Preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        encoding = self.feature_extractor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze() # remove batch dimension
        target = encoding["labels"][0] # remove batch dimension

        return pixel_values, target
