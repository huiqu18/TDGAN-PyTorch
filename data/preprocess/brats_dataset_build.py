
import os
import sys

import h5py
import nibabel as nib
from tqdm import tqdm
import random
import numpy as np


def read_raw_data(dcm_path, save_path, save_name, test_case_number):
    """
    read images and labels from raw data of HGG cases, split the 210 cases into train (170) and test (40),
    and save them in h5 file.

    """
    os.makedirs(save_path, exist_ok=True)
    dcm_folders = sorted(os.listdir(dcm_path))
    result_file = h5py.File(os.path.join(save_path, save_name), "w")

    idx = 0
    for folder in dcm_folders:
        print('[{:d}/{:d}] Processing {:s}'.format(idx+1, len(dcm_folders), folder))
        if idx < test_case_number:
            type = "test"
        else:
            type = "train"
        if os.path.isdir(os.path.join(dcm_path, folder)):
            sys.stdout.flush()
            flair = nib.load(os.path.join(dcm_path, folder, f"{folder}_flair.nii.gz")).get_fdata()
            seg = nib.load(os.path.join(dcm_path, folder, f"{folder}_seg.nii.gz")).get_fdata()
            t1 = nib.load(os.path.join(dcm_path, folder, f"{folder}_t1.nii.gz")).get_fdata()
            t1ce = nib.load(os.path.join(dcm_path, folder, f"{folder}_t1ce.nii.gz")).get_fdata()
            t2 = nib.load(os.path.join(dcm_path, folder, f"{folder}_t2.nii.gz")).get_fdata()

            result_file.create_dataset(f"{type}/{idx}/flair", data=flair)
            db = result_file.create_dataset(f"{type}/{idx}/seg", data=seg)
            result_file.create_dataset(f"{type}/{idx}/t1", data=t1)
            result_file.create_dataset(f"{type}/{idx}/t1ce", data=t1ce)
            result_file.create_dataset(f"{type}/{idx}/t2", data=t2)

            db.attrs['id'] = folder
            idx += 1

    result_file.close()


def get_training_data(h5_filepath, save_filepath, modality):
    def _range_idx(label, margin):
        uniq = []
        for i in range(label.shape[2]):
            if len(np.unique(label[:, :, i])) > 1:
                uniq.append(1)
            else:
                uniq.append(0)
        min_idx = max(0, uniq.index(1) - margin)
        upper_idx = len(uniq) - uniq[::-1].index(1) - 1
        max_idx = min(len(uniq), upper_idx + margin)
        return min_idx, max_idx

    original_file = h5py.File(h5_filepath, 'r')
    dataset = original_file['train']
    save_file = h5py.File(save_filepath, 'w')
    for key in tqdm(dataset.keys()):
        dcm = dataset[f"{key}/{modality}"][()]
        label = dataset[f"{key}/seg"][()]

        start, end = _range_idx(label, 20)
        for i in range(start, end):
            slice_dcm = dcm[:, :, i]
            slice_dcm = slice_dcm * ((pow(2, 8) - 1) / slice_dcm.max())
            slice_dcm = np.repeat(slice_dcm[..., np.newaxis], 3, axis=2)
            slice_dcm = slice_dcm.astype('uint8')
            slice_label = label[:, :, i].astype('uint8')
            
            save_file.create_dataset('images/{:s}_{:d}'.format(key, i), data=slice_dcm)
            save_file.create_dataset('labels/{:s}_{:d}'.format(key, i), data=slice_label)
    print('number of images:', len(save_file['images']))
    original_file.close()
    save_file.close()


def remove_empty_slices(h5_filepath, save_filepath):
    ori_h5 = h5py.File(h5_filepath, 'r')
    new_h5 = h5py.File(save_filepath, 'w')
    for key in tqdm(ori_h5['images'].keys()):
        image = ori_h5['images/{:s}'.format(key)][()]
        label = ori_h5['labels/{:s}'.format(key)][()]

        if np.count_nonzero(label) < 10:
            continue

        new_h5.create_dataset('images/{:s}'.format(key), data=image)
        new_h5.create_dataset('labels/{:s}'.format(key), data=label)
    print(len(ori_h5['images'].keys()))
    print(len(new_h5['images'].keys()))
    ori_h5.close()
    new_h5.close()


def select_10_percent_data(h5_filepath, save_filepath, modality):
    ori_h5 = h5py.File(h5_filepath, 'r')
    new_h5 = h5py.File(save_filepath, 'w')

    keys = list(ori_h5['images'].keys())
    patients = [x.split('_')[0] for x in keys]
    patients = np.unique(patients)
    random.seed(1)
    random.shuffle(patients)
    if modality == 't2':
        patients_part = patients[:int(0.1*len(patients))]
    elif modality == 't1':
        patients_part = patients[int(0.2*len(patients)):int(0.3*len(patients))]
    else:
        raise ValueError('Modality should be either t2 or t1.')

    for key in keys:
        if key.split('_')[0] in patients_part:
            new_h5.create_dataset('images/{:s}'.format(key), data=ori_h5['images/{:s}'.format(key)][()])
            new_h5.create_dataset('labels/{:s}'.format(key), data=ori_h5['labels/{:s}'.format(key)][()])

    print('number of files in original dataset: {:d}'.format(len(keys)))
    print('number of files in new dataset: {:d}'.format(len(new_h5['images'].keys())))
    ori_h5.close()
    new_h5.close()


# read images and labels from raw data and save in h5 file
read_raw_data('/share_hd1/db/BRATS/2018/HGG', "/share_hd1/db/BRATS/2018", "BraTS18.h5", 40)

# process the dataset for t2 modality
os.makedirs('../../db/brain/original', exist_ok=True)
get_training_data('/share_hd1/db/BRATS/2018/BraTS18.h5',  '../../db/brain/original/train_brats.h5', modality='t2')
remove_empty_slices('../../db/brain/original/train_brats.h5',  '../../db/brain/train_brats.h5')
select_10_percent_data( '../../db/brain/train_brats.h5', '../../db/brain/train_brats_0.1.h5', modality='t2')

# process the dataset for t1 modality
get_training_data('/share_hd1/db/BRATS/2018/BraTS18.h5',  '../../db/brain/original/train_brats_t1.h5', modality='t1')
remove_empty_slices('../../db/brain/original/train_brats_t1.h5',  '../../db/brain/train_brats_t1.h5')
select_10_percent_data( '../../db/brain/train_brats_t1.h5', '../../db/brain/train_brats_t1_0.1.h5', modality='t1')