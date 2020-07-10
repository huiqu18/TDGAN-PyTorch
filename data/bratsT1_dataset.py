import os.path
import random
import sys

import cv2
import h5py
import numpy as np
import scipy.ndimage as ndimage
from PIL import Image

from data.base_dataset import BaseDataset, get_params, get_transform


class BratsT1Dataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """

        h5_name = "train_brats_t1_0.1.h5"

        print(f"Load: {h5_name}")
        self.is_test = False
        self.real_tumor = False
        self.extend_len = 0
        self.multi_label = True
        BaseDataset.__init__(self, opt)
        self.brats_file = h5py.File(os.path.join(opt.dataroot, h5_name), 'r')
        if 'train' in self.brats_file:
            train_db = self.brats_file['train']
        else:
            train_db = self.brats_file
        self.dcm, self.label, self.seg_label = self.build_pairs(train_db)

        # self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        # self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert (self.opt.load_size >= self.opt.crop_size)  # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def build_pairs(self, dataset):
        dcm_arr = []
        label_arr = []
        seg_label_arr = []
        images = dataset['images']
        labels = dataset['labels']

        keys = images.keys()
        for key in keys:
            img = images[key][()]
            label = labels[key][()]

            seg_label = (label > 0).astype(np.uint8) * 255

            # label = label * (250 / (label.max() + 1e-8))  # multi-label
            label = (label > 0).astype(np.uint8) * 255  # single label
            label = self.merge_skull(img[:, :, 0], label, default_skull_value=255)
            # label = label[:, :, None].repeat(3, axis=2)

            dcm_arr.append(img)
            label_arr.append(label)
            seg_label_arr.append(seg_label)

        return dcm_arr, label_arr, seg_label_arr


    def seg_in_skull(self, seg, mask):
        # ndimage.binary_fill_holes(skull_mask)
        seg = mask * seg
        return seg

    def merge_skull(self, skull_mask, slice_label, default_skull_value=5):
        # Add skull structure into label
        skull_mask = ndimage.binary_fill_holes(skull_mask)
        skull_mask = cv2.Laplacian(skull_mask.astype("uint8"), cv2.CV_8U)
        skull_mask[skull_mask > 0] = default_skull_value
        slice_label = slice_label + skull_mask

        # slice_label = slice_label * (255 / (slice_label.max() + 1e-8))
        # slice_label = slice_label.astype("uint8")

        # slice_label[slice_label > 0] = 255

        return slice_label

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        A = self.label[index]
        B = self.dcm[index]
        seg = self.seg_label[index]
        A = Image.fromarray(A).convert('RGB')
        B = Image.fromarray(B).convert('RGB')
        seg = Image.fromarray(seg).convert('RGB')

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        if self.opt.phase.lower() != 'train':
            transform_params['crop_pos'] = (0, 0)
            transform_params['vflip'] = False
            transform_params['hflip'] = False

        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1), method=Image.NEAREST)
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))
        seg_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1), method=Image.NEAREST)

        A = A_transform(A)
        B = B_transform(B)
        seg = seg_transform(seg)

        return {'A': A, 'B': B, 'A_paths': str(index), 'B_paths': str(index), 'Seg_label': seg}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.dcm)
