# -*- coding: utf-8 -*-

import os
import math
import random
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter

import torch
import torchmetrics
from torchmetrics import Accuracy
from torch.utils.data import DataLoader

from PIL import Image, ImageEnhance
from sklearn.metrics import precision_recall_fscore_support as f_score

from transformers import Trainer
from transformers import default_data_collator
from transformers.training_args import TrainingArguments
from transformers.feature_extraction_utils import FeatureExtractionMixin

from tqdm import tqdm
from tabulate import tabulate

from hugsvision.dataio.ImageClassificationCollator import ImageClassificationCollator

class VisionClassifierTrainer:

  """
  🤗 Constructor for the image classifier trainer
  """
  def __init__(
    self,
    ids2labels,
    output_dir        :str,
    model_name        :str,
    model             :torch.nn.Module,
    feature_extractor :FeatureExtractionMixin,
    dataset           :torch.utils.data.Dataset,
    max_epochs    = 1,
    cores         = 4,
    batch_size    = 8,
    test_ratio    = 0.15,
    lr            = 2e-5,
    eval_metric   = "accuracy",
    fp16          = True,
    shuffle       = True,
    balanced      = False,
    augmentation  = False,
  ):

    self.model_name        = model_name
    self.dataset           = dataset
    self.output_dir        = output_dir
    self.lr                = lr
    self.batch_size        = batch_size
    self.max_epochs        = max_epochs
    self.shuffle           = shuffle
    self.test_ratio        = test_ratio
    self.model             = model
    self.feature_extractor = feature_extractor
    self.cores             = cores
    self.fp16              = fp16
    self.eval_metric       = eval_metric
    self.ids2labels        = ids2labels
    self.balanced          = balanced
    self.augmentation      = augmentation

    # Processing device (CPU / GPU)
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Setup the metric
    self.metric = torchmetrics.Accuracy()
    
    # Get the classifier collator
    self.collator = ImageClassificationCollator(self.feature_extractor)

    # Get the model output path
    self.output_path = self.__getOutputPath()
    self.logs_path   = self.output_path
    
    # Open the logs file
    self.__openLogs()

    # Split and convert to dataloaders
    self.train, self.test = self.__splitDatasets()

    """
    🏗️ Build the trainer
    """
    self.training_args = TrainingArguments(
        output_dir                  = self.output_path,
        save_total_limit            = 2,
        weight_decay                = 0.01,
        save_steps                  = 10000,
        learning_rate               = self.lr,
        per_device_train_batch_size = self.batch_size,
        per_device_eval_batch_size  = self.batch_size,
        num_train_epochs            = self.max_epochs,
        metric_for_best_model       = self.eval_metric,
        logging_dir                 = self.logs_path,
        evaluation_strategy         = "epoch",
        load_best_model_at_end      = False,
        overwrite_output_dir        = True,
        fp16=self.fp16,
    )
    
    self.trainer = Trainer(
      self.model,
      self.training_args,
      train_dataset = self.train,
      eval_dataset  = self.test,
      data_collator = self.collator,
    )
    print("Trainer builded!")

    """
    ⚙️ Train the given model on the dataset
    """
    print("Start Training!")
    self.trainer.train()
    self.trainer.save_model(self.output_path + "/trainer/")
    self.model.save_pretrained(self.output_path + "/model/")
    self.feature_extractor.save_pretrained(self.output_path + "/feature_extractor/")
    print("Model saved at: \033[93m" + self.output_path + "\033[0m")

    # Close the logs file
    self.logs_file.close()

  """
  📜 Open the logs file
  """
  def __openLogs(self):    

    # Open the logs file
    self.logs_file = open(self.logs_path + "/logs.txt", "a")

  """
  📍 Get the path of the output model
  """
  def __getOutputPath(self):

    path = os.path.join(
      self.output_dir,
      self.model_name.upper() + "/" + str(self.max_epochs) + "_" + datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
    )

    # Create the full path if doesn't exist yet
    if not os.path.isdir(path):
        os.makedirs(path)

    return path

  """
  🧬 Apply data augmentation on the input image
  Source: https://medium.com/lunit/photometric-data-augmentation-in-projection-radiography-bed3ae9f55c3
  """
  def __augmentation(self, image, beta=0.33):

    # Random augmentation
    if random.randint(0,100) < (beta*100):
    
      # Random Contrast
      im3 = ImageEnhance.Contrast(image)
      im3.enhance(random.uniform(0.5, 1.0)).show()
      
      # Random Noise

    return image

  """
  ⚖️ Balance the dataset according to the less represented label
  """
  def __balance(self, train_ds, train_classes):

    # Get the less represented label in train
    less_represented_train = min(train_classes)

    labels = {}

    # For each image
    for img, label in train_ds:

      # Create the label array if isn't
      if label not in labels:
        labels[label] = []
      
      # Add the image to the array
      labels[label].append(img)
    
    # New dataset
    balanced_ds = []

    # For each label
    for label in labels:

      # Get images
      imgs = labels[label]

      # For each image
      for img in imgs[0:less_represented_train]:

        # Create a tuple: image and label 
        t = (img, label)

        # Add it
        balanced_ds.append(t)

    print("The less represented label in train as " + str(less_represented_train) + " occurrences")
    print("Size of train after balancing is " + str(len(balanced_ds)))

    return balanced_ds
    
  """
  ✂️ Split the dataset into sub-datasets
  """
  def __splitDatasets(self):

    print("Split Datasets...")
    
    indices = torch.randperm(len(self.dataset)).tolist()

    # Index of the validation corpora
    train_index = math.floor(len(indices) * (1 - self.test_ratio))

    # TRAIN
    train_ds = torch.utils.data.Subset(self.dataset, indices[:train_index])
    ct_train = Counter([label for _, label in train_ds])
    train_classes = [ct_train[a] for a in sorted(ct_train)]
    
    # If balanced is enabled
    if self.balanced == True:

      # Balance the train dataset
      train_ds = self.__balance(train_ds, train_classes)
      
      # Compute again the stats
      ct_train = Counter([label for _, label in train_ds])
      train_classes = [ct_train[a] for a in sorted(ct_train)]

    # If data augmentation is enabled
    if self.augmentation == True:

      new_ds = []

      # For each annotated image
      for img, label in train_ds:

        # Augment it
        new_ds.append(
          (self.__augmentation(img), label)
        )
      
      # Replace by the augmented data
      train_ds = new_ds

    # TEST
    test_ds = torch.utils.data.Subset(self.dataset, indices[train_index:])
    ct_test = Counter([label for _, label in test_ds])
    test_classes = [ct_test[a] for a in sorted(ct_test)]

    table_repartition = [
      ["Train"] + train_classes + [str(len(train_ds))],
      ["Test"] + test_classes + [str(len(test_ds))],
    ]

    repartitions_table = tabulate(table_repartition, ["Dataset"] + list(self.ids2labels.values()) + ["Total"], tablefmt="pretty")
    print(repartitions_table)

    # Write logs
    self.logs_file.write(repartitions_table + "\n")

    return train_ds, test_ds

  """
  🧪 Evaluate the performances of the system of the test sub-dataset given a f1-score
  """
  def evaluate_f1_score(self):

    # Get the hypothesis and predictions
    all_target, all_preds = self.__evaluate()

    # Get the labels as a list
    labels = list(self.ids2labels.keys())

    # Compute f-score
    precision, recall, fscore, support = f_score(all_target, all_preds)

    table_f_score = []

    # Add macro scores for each classes
    for i in range(len(labels)):
        table_f_score.append(
          [
              self.ids2labels[labels[i]],
              str(round(precision[i] * 100,2)) + " %",
              str(round(recall[i] * 100,2)) + " %",
              str(round(fscore[i] * 100,2)) + " %",
              support[i],
          ]
        )

    # Add global macro scores
    table_f_score.append(
      [
          "Macro",
          str(round((sum(precision) / len(precision)) * 100, 2)) + " %",
          str(round((sum(recall)    / len(recall))    * 100, 2)) + " %",
          str(round((sum(fscore)    / len(fscore))    * 100, 2)) + " %",
          str(sum(support)),
      ]
    )

    # Print precision, recall, f-score and support for each classes
    f_score_table = tabulate(table_f_score, tablefmt="psql", headers=["Label", "Precision", "Recall", "F-Score", "Support"])
    print(f_score_table)

    # Write logs
    self.__openLogs()
    self.logs_file.write(f_score_table + "\n")
    self.logs_file.close()

    print("Logs saved at: \033[93m" + self.logs_path + "\033[0m")

    return all_target, all_preds

  """
  🧪 Evaluate the performances of the system of the test sub-dataset for the chosen evaluation metric
  """
  def evaluate(self):
    return self.trainer.evaluate()

  """
  🧪 Evaluate the performances of the system of the test sub-dataset
  """
  def __evaluate(self):
        
    all_preds  = []
    all_target = []

    # For each image
    for image, label in tqdm(self.test):

        # Compute
        inputs  = self.feature_extractor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        # Get predictions from the softmax layer
        preds = outputs.logits.softmax(1).argmax(1).tolist()
        all_preds.extend(preds)

        # Get hypothesis
        all_target.append(label)

    return all_preds, all_target

  """
  🧪 Test on a single image
  """
  def testing(self, img,expected):
    image_array = Image.open(img)
    inputs      = self.feature_extractor(images=image_array, return_tensors="pt").to(self.device)
    outputs     = self.model(**inputs)
    preds       = outputs.logits.softmax(1).argmax(1).tolist()[0]
    print(
      "Predicted class: ",
      self.ids2labels[str(preds)],
      "(", str(preds), " - ", self.ids2labels[str(expected)], ") ",
      str(preds == expected)
    )
    return preds