import torch
import pytorch_lightning as pl
from transformers import DetrConfig, DetrForObjectDetection

class Detr(pl.LightningModule):

    """
    ⚡ Constructor for the DETR Object Detection Model
    Warning: Currently support only DETR
    """
    def __init__(
        self,
        lr,
        lr_backbone,
        weight_decay,
        id2label,
        label2id,
        train_dataloader,
        val_dataloader,
        model_path = "facebook/detr-resnet-50"
    ):
        super().__init__()

        self.train_dl = train_dataloader
        self.val_dl   = val_dataloader
        
        # Load the base model
        model      = DetrForObjectDetection.from_pretrained(model_path)
        state_dict = model.state_dict()

        # Remove class weights
        del state_dict["class_labels_classifier.weight"]
        del state_dict["class_labels_classifier.bias"]

        # Define new model with custom class classifier
        model = DetrForObjectDetection(DetrConfig.from_pretrained(
            model_path,
            num_labels=len(id2label),
            id2label   = id2label,
            label2id   = label2id,
        ))
        model.load_state_dict(state_dict, strict=False)
        self.model = model

        # Learning Rate
        # See https://github.com/PyTorchLightning/pytorch-lightning/pull/1896
        self.lr           = lr
        self.lr_backbone  = lr_backbone
        self.weight_decay = weight_decay

    def forward(self, pixel_values, pixel_mask):

        return self.model(
            pixel_values = pixel_values,
            pixel_mask   = pixel_mask
        )
    
    def common_step(self, batch, batch_idx):

        # Get image
        pixel_values = batch["pixel_values"]
        pixel_mask   = batch["pixel_mask"]

        # Get labels
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        # Predict
        outputs = self.model(
            pixel_values = pixel_values,
            pixel_mask   = pixel_mask,
            labels       = labels,
        )

        # Loss
        loss      = outputs.loss
        loss_dict = outputs.loss_dict

        return loss, loss_dict

    def training_step(self, batch, batch_idx):

        loss, loss_dict = self.common_step(batch, batch_idx)   

        # logs metrics for each training_step, and the average across the epoch
        self.log("training_loss", loss)

        for k,v in loss_dict.items():
            self.log("train_" + k, v.item())

        return loss

    def validation_step(self, batch, batch_idx):

        loss, loss_dict = self.common_step(batch, batch_idx)

        self.log("validation_loss", loss)

        for k,v in loss_dict.items():
            self.log("validation_" + k, v.item())

        return loss

    def configure_optimizers(self):

        param_dicts = [
            {
                "params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]
            },
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.lr_backbone,
            },
        ]

        optimizer = torch.optim.AdamW(
            param_dicts,
            lr           = self.lr,
            weight_decay = self.weight_decay
        )
        
        return optimizer

    def train_dataloader(self):
        return self.train_dl

    def val_dataloader(self):
        return self.val_dl
