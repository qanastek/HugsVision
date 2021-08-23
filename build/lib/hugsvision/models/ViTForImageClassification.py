import torch.nn as nn

from transformers import ViTModel
from transformers.modeling_outputs import SequenceClassifierOutput

class ViTForImageClassification(nn.Module):

    def __init__(self, model):
        super(ViTForImageClassification, self).__init__()

        # Load the huggingface model
        self.vit = model
        # Drop 10 % of the solutions
        self.dropout = nn.Dropout(0.1)
        # Adapt the softmax layer to the model
        self.classifier = nn.Linear(
            self.vit.config.hidden_size,
            len(self.vit.config.id2label)
        )
        # The size of the softmax layer
        self.num_labels = len(self.vit.config.id2label)

    def forward(self, pixel_values, labels):

        # Predict
        outputs = self.vit(pixel_values=pixel_values)

        # Find the corresponding label
        output = self.dropout(outputs.last_hidden_state[:,0])
        logits = self.classifier(output)

        # Compute loss
        loss = None
        if labels is not None:
          loss_fct = nn.CrossEntropyLoss()
          loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # Return forward results
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )