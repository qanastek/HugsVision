# -*- coding: utf-8 -*-

import os
import math
import random
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter

import torch
from torchvision.datasets import ImageFolder

from PIL import Image, ImageEnhance

from tabulate import tabulate

class VisionDataset:

    """
    🧬 Apply data augmentation on the input image
    Source: https://medium.com/lunit/photometric-data-augmentation-in-projection-radiography-bed3ae9f55c3
    """
    @staticmethod
    def __augmentation(image, beta=0.33):

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
    @staticmethod
    def __balance(train_ds, train_classes):

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
    def splitDatasets(dataset, id2label, test_ratio=0.15, balanced=True, augmentation=False):

        print("Split Datasets...")
        
        indices = torch.randperm(len(dataset)).tolist()

        # Index of the validation corpora
        train_index = math.floor(len(indices) * (1 - test_ratio))

        # TRAIN
        train_ds = torch.utils.data.Subset(dataset, indices[:train_index])
        ct_train = Counter([label for _, label in train_ds])
        train_classes = [ct_train[a] for a in sorted(ct_train)]
        
        # If balanced is enabled
        if balanced == True:

            # Balance the train dataset
            train_ds = VisionDataset.__balance(train_ds, train_classes)
            
            # Compute again the stats
            ct_train = Counter([label for _, label in train_ds])
            train_classes = [ct_train[a] for a in sorted(ct_train)]

        # If data augmentation is enabled
        if augmentation == True:

            new_ds = []

            # For each annotated image
            for img, label in train_ds:

                # Augment it
                new_ds.append(
                    (VisionDataset.__augmentation(img), label)
                )
            
            # Replace by the augmented data
            train_ds = new_ds

        # TEST
        test_ds = torch.utils.data.Subset(dataset, indices[train_index:])
        ct_test = Counter([label for _, label in test_ds])
        test_classes = [ct_test[a] for a in sorted(ct_test)]

        table_repartition = [
            ["Train"] + train_classes + [str(len(train_ds))],
            ["Test"] + test_classes + [str(len(test_ds))],
        ]

        repartitions_table = tabulate(table_repartition, ["Dataset"] + list(id2label.values()) + ["Total"], tablefmt="pretty")
        print(repartitions_table)

        # # Write logs
        # self.logs_file.write(repartitions_table + "\n")

        return torch.utils.data.Subset(train_ds, list(range(0,len(train_ds)))), torch.utils.data.Subset(test_ds, list(range(0,len(test_ds))))

    @staticmethod
    def fromImageFolder(dataset:str, test_ratio=0.15, balanced=True, augmentation=False):
        
        # Create ImageFolder from path
        dataset = ImageFolder(dataset)

        # Both way indexes
        label2id, id2label = VisionDataset.getConfig(dataset)

        # Split
        train, test = VisionDataset.splitDatasets(dataset,id2label, test_ratio, balanced, augmentation)

        return train, test, id2label, label2id

    @staticmethod
    def fromImageFolders(train:str, test:str):
        
        # Split
        train = ImageFolder(train)
        test  = ImageFolder(test)

        # Both way indexes
        label2id, id2label = VisionDataset.getConfig(train)

        # Build Subsets
        train = torch.utils.data.Subset(train,list(range(0,len(train))))
        test  = torch.utils.data.Subset(test,list(range(0,len(test))))

        return train, test, id2label, label2id

    @staticmethod
    def getConfig(dataset:torch.utils.data.Dataset):

        label2id = {}
        id2label = {}

        for i, class_name in enumerate(dataset.classes):
            label2id[class_name] = str(i)
            id2label[str(i)] = class_name

        return label2id, id2label
