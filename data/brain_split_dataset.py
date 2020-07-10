import os.path
import random
import sys

import h5py
import numpy as np
import scipy.ndimage as ndimage
from PIL import Image

from data.base_dataset import BaseDataset, get_params, get_transform
from data.brats_dataset import BratsDataset
from data.nuclei_dataset import NucleiDataset


class BrainSplitDataset(BaseDataset):

    def __init__(self, opt):
        self.split_db = [NucleiDataset(opt, 4),
                         BratsDataset(opt, 1),
                         BratsDataset(opt, 2)]

    def __getitem__(self, index):

        result = {}
        for k, v in enumerate(self.split_db):
            database = v
            if index >= len(database):
                index = index % len(database)

            index_value = database[index]
            result['A_' + str(k)] = index_value['A']
            result['B_' + str(k)] = index_value['B']
            result['A_paths_' + str(k)] = index_value['A_paths']
            result['B_paths_' + str(k)] = index_value['B_paths']
            result['Seg_label_' + str(k)] = index_value['Seg_label']
        return result

    def __len__(self):
        """Return the total number of images in the dataset."""
        length = 0
        for i in self.split_db:
            if len(i) > length:
                length = len(i)

        return length

