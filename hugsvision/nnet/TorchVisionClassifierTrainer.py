# -*- coding: utf-8 -*-

import os
import json
import math
import random
import argparse
from pathlib import Path
import multiprocessing as mp
from datetime import datetime
from collections import Counter

import torch
import torchmetrics
import torch.nn as nn
from torchmetrics import Accuracy
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR

from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from tqdm import tqdm
from tabulate import tabulate
from PIL import Image, ImageEnhance

class TorchVisionClassifierTrainer:

  """
  🤗 Constructor for the image classifier trainer
  """
  def __init__(
    self,
    id2label,
    label2id,
    model_name        :str,
    train             :torch.utils.data.Dataset,
    test              :torch.utils.data.Dataset,
    output_dir    = "./models",
    max_epochs    = 1,
    workers       = mp.cpu_count(),
    batch_size    = 8,
    lr            = 2e-5,
    weight_decay  = 0.01,
    momentum      = 0.9,
    gamma         = 0.96,
    pretrained    = True,
    force_cpu     = False,
  ):

    self.train             = train
    self.test              = test
    self.output_dir        = output_dir if output_dir.endswith("/") else output_dir + "/"
    self.lr                = lr
    self.weight_decay      = weight_decay
    self.momentum          = momentum
    self.gamma             = gamma
    self.batch_size        = batch_size
    self.max_epochs        = max_epochs
    self.workers           = workers
    self.model_name        = model_name
    self.pretrained        = pretrained
    self.ids2labels        = id2label
    self.labels2ids        = label2id
    self.best_acc          = 0
    self.logs_path         = self.output_dir.upper() + "logs/"
    self.config_path       = self.output_dir.upper() + "config.json"
    self.current_date      = datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
    self.config            = {}

    self.data_loader_train = torch.utils.data.DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=self.workers)
    self.data_loader_test = torch.utils.data.DataLoader(self.test, batch_size=self.batch_size, shuffle=True, num_workers=self.workers)

    print(self.ids2labels)
    print(self.labels2ids)

    # Processing device (CPU / GPU)
    if force_cpu == True:
      self.device = "cpu"
    else:
      self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Setup the metric
    self.metric = torchmetrics.Accuracy()

    self.num_classes = len(self.ids2labels.keys())
    self.config["num_classes"] = self.num_classes
    
    # Open the logs file
    self.__openLogs()

    # Load the model from the TorchVision Models Zoo 
    if pretrained:
      print("Load the pre-trained model " + self.model_name)
      self.model = models.__dict__[self.model_name](pretrained=True)
    else:
      print("Load the model " + self.model_name)
      self.model = models.__dict__[self.model_name]()

    print(self.model)
    
    # """
    # 🏗️ Build the model
    # """
    if self.model_name.startswith('alexnet') or self.model_name.startswith('vgg'):
      self.model.classifier[6] = nn.Linear(self.model.classifier[6].in_features, self.num_classes)
      self.model.features = torch.nn.DataParallel(self.model.features)
      self.model.to(self.device)
      self.config["hidden_size"] = self.model.classifier[6].in_features
    else:
      if self.model_name.startswith('resnet') or self.model_name.startswith('googlenet') or self.model_name.startswith('resnext') or self.model_name.startswith('shufflenet_v2') or self.model_name.startswith('wide_resnet'):
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
        self.config["hidden_size"] = self.model.fc.in_features

      elif self.model_name.startswith('squeezenet'):
        self.model.classifier[1] = nn.Conv2d(512, self.num_classes, kernel_size=(1,1), stride=(1,1))
        self.config["hidden_size"] = 512

      elif self.model_name.startswith('densenet'):
        self.model.classifier = nn.Linear(self.model.classifier.in_features, self.num_classes)
        self.config["hidden_size"] = self.model.classifier.in_features

      elif self.model_name.startswith('inception_v3'):
        self.model.AuxLogits.fc = nn.Linear(self.model.AuxLogits.fc.in_features, self.num_classes)
        self.model.fc = nn.Linear(2048, self.num_classes)
        self.config["hidden_size"] = self.model.AuxLogits.fc.in_features

      elif self.model_name.startswith('mobilenet_v2') or self.model_name.startswith('mnasnet'):
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, self.num_classes)
        self.config["hidden_size"] = self.model.classifier[1].in_features

      elif self.model_name.startswith('mobilenet_v3'):
        self.model.classifier[3] = nn.Linear(self.model.classifier[3].in_features, self.num_classes)
        self.config["hidden_size"] = self.model.classifier[3].in_features

      self.model = torch.nn.DataParallel(self.model).to(self.device)

    archi = ""
    archi += "="*50 + "\n"
    archi += "Model architecture:" + "\n"
    archi += "="*50 + "\n"
    archi += str(self.model) + "\n"
    archi += "="*50 + "\n"
    print(archi)
    
    # Write in logs
    self.__openLogs()
    self.logs_file.write(archi + "\n")
    self.logs_file.close()

    self.criterion = nn.CrossEntropyLoss().to(self.device)
    self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
    self.scheduler = ExponentialLR(self.optimizer, gamma=self.gamma)

    """
    ⚙️ Train the given model on the dataset
    """
    print("Start Training!")
    self.training()

    # Close the logs file
    self.logs_file.close()

    self.config["id2label"] = self.ids2labels
    self.config["label2id"] = self.labels2ids
    self.config["architectures"] = [self.model_name]
    
    with open(self.config_path, 'w') as json_file:
      json.dump(self.config, json_file, indent=4)

  """
  📜 Open the logs file
  """
  def __openLogs(self):

    # Check if the directory already exist
    os.makedirs(self.logs_path, exist_ok=True)

    # Open the logs file
    self.logs_file = open(self.logs_path + "logs_" + self.current_date + ".txt", "a")

  """
  🧪 Evaluate the performances of the system of the test sub-dataset given a f1-score
  """
  def evaluate_f1_score(self):

    # Get the hypothesis and predictions
    all_target, all_preds = self.evaluate()

    table = metrics.classification_report(
      all_target,
      all_preds,
      labels = [int(a) for a in list(self.ids2labels.keys())],
      target_names = list(self.labels2ids.keys()),
      zero_division = 0,
    )
    print(table)

    # Write logs
    self.__openLogs()
    self.logs_file.write(table + "\n")
    self.logs_file.close()

    print("Logs saved at: \033[93m" + self.logs_path + "\033[0m")

    return all_target, all_preds

  # """
  # 🧪 Test on a single image
  # TODO: Rebuild all
  # """
  # def testing(self, img,expected):
  #   image_array = Image.open(img)
  #   inputs      = self.feature_extractor(images=image_array, return_tensors="pt").to(self.device)
  #   outputs     = self.model(**inputs)
  #   preds       = outputs.logits.softmax(1).argmax(1).tolist()[0]
  #   print(
  #     "Predicted class: ",
  #     self.ids2labels[str(preds)],
  #     "(", str(preds), " - ", self.ids2labels[str(expected)], ") ",
  #     str(preds == expected)
  #   )
  #   return preds
  
  """
  👩‍🎓 Training phase
  """
  def training(self):

    for epoch in tqdm(range(self.max_epochs)):

      # Train the epoch
      self.compute_batches(epoch)

      # Evaluate on validation dataset
      all_targets, all_predictions = self.evaluate()
      total_acc = accuracy_score(all_targets, all_predictions)

      f1_score = classification_report(all_targets, all_predictions, target_names=list(self.ids2labels.values()))
      print(f1_score)

      os.makedirs(self.output_dir.upper(), exist_ok=True)

      if total_acc > self.best_acc:
        filename = self.output_dir.upper() + 'best_model.pth'
      else:
        filename = self.output_dir.upper() + 'last_model.pth'

      self.best_acc = max(total_acc, self.best_acc)

      torch.save(self.model, filename)
      # torch.save({'epoch': epoch + 1, 'arch': self.model_name, 'state_dict': self.model.state_dict(), 'best_prec1': self.best_acc }, filename)
      print("Model saved at: \033[93m" + filename + "\033[0m")

      self.__openLogs()
      self.logs_file.write(f1_score + "\n")
      self.logs_file.close()

  """
  🗃️ Compute epoch batches
  """
  def compute_batches(self, epoch):

    # Switch to train mode
    self.model.train()

    # For each batch
    for i, (input, target) in enumerate(self.data_loader_train):

      input = input.to(self.device)
      target = target.to(self.device)
      
      input_var = torch.autograd.Variable(input)
      target_var = torch.autograd.Variable(target)

      # compute output
      output = self.model(input_var)
      loss = self.criterion(output, target_var)

      # compute gradient and do SGD step
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

      if i % (len(self.data_loader_train) / 10) == 0:
        print('Epoch: ', epoch, ' | Batch: ', i, " | Loss: ", loss.item())

    self.scheduler.step()

  """
  🧪 Evaluate the performances of the system of the test sub-dataset
  """
  def evaluate(self):

    all_predictions = []
    all_targets = []

    # Switch to evaluate mode
    self.model.eval()

    for i, (input, target) in enumerate(self.data_loader_test):

      input = input.to(self.device)
      input_var = torch.autograd.Variable(input, volatile=True)

      output = self.model(input_var).cpu().data.numpy()
      output = [o.argmax() for o in output]

      all_targets.extend(target.tolist())
      all_predictions.extend(output)

    return all_targets, all_predictions