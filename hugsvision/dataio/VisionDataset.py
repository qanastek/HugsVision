# -*- coding: utf-8 -*-

import torch

class VisionDataset:

    @staticmethod
    def getConfig(dataset:torch.utils.data.Dataset):

        labels2ids = {}
        ids2labels = {}

        for i, class_name in enumerate(dataset.classes):
            labels2ids[class_name] = str(i)
            ids2labels[str(i)] = class_name

        return len(labels2ids), labels2ids, ids2labels
