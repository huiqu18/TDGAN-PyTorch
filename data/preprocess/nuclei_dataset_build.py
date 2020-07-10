
import os, json, glob
from skimage import io
from tqdm import tqdm
import numpy as np
import random

import h5py


def build_h5(data_dir, save_dir):
    with open('{:s}/train_test.json'.format(data_dir), 'r') as file:
        data_list = json.load(file)
        train_list, test_list, test2_list = data_list['train'], data_list['testA'], data_list['testB']

    train_file_all = h5py.File('{:s}/train_nulei.h5'.format(save_dir), "w")
    # test_file = h5py.File('{:s}/test_nuclei.h5'.format(save_dir), "w")

    # train imgs are 250 x 250 resolution extracted from original images
    print('Processing all training files')
    _dump_training_files(train_file_all, data_dir, train_list)

    # # test imgs are original 1000 x 1000 resolution
    # print('Processing test files')
    # for img_name in tqdm(test_list):
    #     name = img_name.split('.')[0]
    #     # images
    #     img = io.imread('{:s}/original_images/{:s}.png'.format(data_dir, name))
    #     test_file.create_dataset('images/{:s}'.format(name), data=img)
    #     # labels
    #     label = io.imread('{:s}/labels/{:s}.png'.format(data_dir, name))
    #     test_file.create_dataset('labels/{:s}'.format(name), data=label)
    #     # ternary labels (for segmentation)
    #     label_ternary = io.imread('{:s}/labels_ternary/{:s}_label.png'.format(data_dir, name))
    #     test_file.create_dataset('labels_ternary/{:s}'.format(name), data=label_ternary)
    #     # weight maps (for segmentation)
    #     weight_map = io.imread('{:s}/weight_maps/{:s}_weight.png'.format(data_dir, name))
    #     test_file.create_dataset('weight_maps/{:s}'.format(name), data=weight_map)
    #     # instance labels (for segmentation)
    #     label_instance = io.imread('{:s}/labels_instance/{:s}.png'.format(data_dir, name))
    #     test_file.create_dataset('labels_instance/{:s}'.format(name), data=label_instance)

    train_file_all.close()
    # test_file.close()


def build_h5_diff_organ(data_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    with open('{:s}/train_test.json'.format(data_dir), 'r') as file:
        data_list = json.load(file)
        train_list, test_list, test2_list = data_list['train'], data_list['testA'], data_list['testB']

    train_file_D1 = h5py.File('{:s}/train_liver.h5'.format(save_dir), "w")
    train_file_D2 = h5py.File('{:s}/train_breast.h5'.format(save_dir), "w")
    train_file_D3 = h5py.File('{:s}/train_kidney.h5'.format(save_dir), "w")
    train_file_D4 = h5py.File('{:s}/train_prostate.h5'.format(save_dir), "w")

    liver_file_list = [filename for filename in train_list if 'Liver' in filename]
    breast_file_list = [filename for filename in train_list if 'Breast' in filename]
    kidney_file_list = [filename for filename in train_list if 'Kidney' in filename]
    prostate_file_list = [filename for filename in train_list if 'Prostate' in filename]

    print('Processing subset training files')
    _dump_training_files(train_file_D1, data_dir, liver_file_list)
    _dump_training_files(train_file_D2, data_dir, breast_file_list)
    _dump_training_files(train_file_D3, data_dir, kidney_file_list)
    _dump_training_files(train_file_D4, data_dir, prostate_file_list)

    train_file_D1.close()
    train_file_D2.close()
    train_file_D3.close()
    train_file_D4.close()


def _dump_training_files(h5_file, data_dir, img_names):
    for img_name in tqdm(img_names):
        name = img_name.split('.')[0]
        for i in range(16):   # 16 patches for each large image
            # images
            img = io.imread('{:s}/patches_256/original_images/{:s}_{:d}.png'.format(data_dir, name, i))
            h5_file.create_dataset('images/{:s}_{:d}'.format(name, i), data=img)
            # labels
            label = io.imread('{:s}/patches_256/labels/{:s}_{:d}.png'.format(data_dir, name, i))
            h5_file.create_dataset('labels/{:s}_{:d}'.format(name, i), data=label)
            # ternary labels (for segmentation)
            label_ternary = io.imread('{:s}/patches_256/labels_ternary/{:s}_{:d}_label.png'.format(data_dir, name, i))
            h5_file.create_dataset('labels_ternary/{:s}_{:d}'.format(name, i), data=label_ternary)
            # weight maps (for segmentation)
            weight_map = io.imread('{:s}/patches_256/weight_maps/{:s}_{:d}_weight.png'.format(data_dir, name, i))
            h5_file.create_dataset('weight_maps/{:s}_{:d}'.format(name, i), data=weight_map)


def split_patches(data_dir, save_dir, post_fix=None):
    import math
    """ split large image into small patches """
    os.makedirs(save_dir, exist_ok=True)

    image_list = os.listdir(data_dir)
    for image_name in image_list:
        name = image_name.split('.')[0]
        if post_fix and name[-len(post_fix):] != post_fix:
            continue
        image_path = os.path.join(data_dir, image_name)
        image = io.imread(image_path)
        seg_imgs = []

        # split into 16 patches of size 250x250
        h, w = image.shape[0], image.shape[1]
        patch_size = 256
        h_overlap = math.ceil((4 * patch_size - h) / 3)
        w_overlap = math.ceil((4 * patch_size - w) / 3)
        for i in range(0, h-patch_size+1, patch_size-h_overlap):
            for j in range(0, w-patch_size+1, patch_size-w_overlap):
                if len(image.shape) == 3:
                    patch = image[i:i+patch_size, j:j+patch_size, :]
                else:
                    patch = image[i:i + patch_size, j:j + patch_size]
                seg_imgs.append(patch)

        for k in range(len(seg_imgs)):
            if post_fix:
                io.imsave('{:s}/{:s}_{:d}_{:s}.png'.format(save_dir, name[:-len(post_fix)-1], k, post_fix), seg_imgs[k])
            else:
                io.imsave('{:s}/{:s}_{:d}.png'.format(save_dir, name, k), seg_imgs[k])


def extract_imgs(h5_file_path, save_path):
    h5_file = h5py.File(h5_file_path, 'r')
    for key in h5_file.keys():
        os.makedirs('{:s}/{:s}'.format(save_path, key), exist_ok=True)
        count = 0
        for file_key in list(h5_file[key].keys())[:1000]:
            img = h5_file['{:s}/{:s}'.format(key, file_key)][()]
            # if key == 'images':
            #     img = (img + 1) / 2.0 * 255
            #     img = img.astype(np.uint8)
            # if len(img.shape) > 2:
            #     img = np.swapaxes(np.swapaxes(img, 0, 1), 1, 2)
            if key == 'labels':
                img = (img > 0).astype(np.uint8) * 255
                if np.count_nonzero(img) < 10:
                    count += 1
            io.imsave('{:s}/{:s}/{:s}.png'.format(save_path, key, file_key), img)
        print(count)


split_patches('../../db/nuclei/original/original_images', '../../db/nuclei/original/patches_256/original_images')
split_patches('../../db/nuclei/original/labels', '../../db/nuclei/original/patches_256/labels')
split_patches('../../db/nuclei/original/labels_ternary', '../../db/nuclei/original/patches_256/labels_ternary', post_fix='label')
split_patches('../../db/nuclei/original/weight_maps', '../../db/nuclei/original/patches_256/weight_maps', post_fix='weight')

build_h5('../../db/nuclei/original', '../../db/nuclei/for_gan_training')
build_h5_diff_organ('../../db/nuclei/original', '../../db/nuclei/for_gan_training')

