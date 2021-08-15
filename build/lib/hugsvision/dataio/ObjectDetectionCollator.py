"""
📁 Object Detection Collator
"""
class ObjectDetectionCollator:
    
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor
 
    def __call__(self, batch):

        # Get all the pixel matrix
        pixel_values = [item[0] for item in batch]

        # Create a mask of the biggest image
        encoding = self.feature_extractor.pad_and_create_pixel_mask(pixel_values, return_tensors="pt")

        # Get all the labels
        labels = [item[1] for item in batch]

        # Convert the data to the DETR format
        batch = {}
        batch['pixel_values'] = encoding['pixel_values']
        batch['pixel_mask'] = encoding['pixel_mask']
        batch['labels'] = labels

        return batch
