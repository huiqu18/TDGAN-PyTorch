"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
import sys

import h5py
import numpy as np
from tqdm import tqdm
from PIL import Image
from skimage import io

from data import create_dataset
from models import create_model
from options.test_options import TestOptions
from util import html
from util.visualizer import save_images


def save_brain_data(gt, visuals, index, model, file, dataset_num, seg_label, append_idx, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    real_img = visuals['real_B'][index]
    syn_img = visuals['fake_B'][index]
    label = visuals['real_A'][index]
    seg_label = seg_label[index]

    img_path = model.get_image_paths()  # get image paths

    syn_img = syn_img.cpu().numpy()
    syn_img = (syn_img + 1) * (255 / 2)
    syn_img = syn_img.astype("uint8")
    syn_img = np.moveaxis(syn_img, 0, -1)

    label = label.cpu().numpy()
    label = (label + 1) * (255 / 2)
    label = label.astype("uint8")
    label = np.moveaxis(label, 0, -1)

    seg_label = seg_label.cpu().numpy()
    seg_label = (seg_label + 1) * (255 / 2)
    seg_label = seg_label.astype("uint8")
    seg_label = np.moveaxis(seg_label, 0, -1)

    real_img = real_img.cpu().numpy()
    real_img = (real_img + 1) * (255 / 2)
    real_img = real_img.astype("uint8")
    real_img = np.moveaxis(real_img, 0, -1)

    # # # for brats, resize back to 240x240
    # if dataset_num > 0: # brats
    #     size = (240, 240)
    #     syn_img = Image.fromarray(syn_img).resize(size)
    #     label = Image.fromarray(label).resize(size, Image.NEAREST)
    #     seg_label = Image.fromarray(seg_label).resize(size, Image.NEAREST)
    #     real_img = Image.fromarray(real_img).resize(size)

    # write to the h5 file
    file.create_dataset(f"images/D{dataset_num}_{img_path[0]}_{index}_{append_idx}", data=syn_img)
    file.create_dataset(f"labels/D{dataset_num}_{img_path[0]}_{index}_{append_idx}", data=label)
    file.create_dataset(f"seg_labels/D{dataset_num}_{img_path[0]}_{index}_{append_idx}", data=seg_label)
    file.create_dataset(f"real_images/D{dataset_num}_{img_path[0]}_{index}_{append_idx}", data=real_img)

    # save as png image file for better visualization: [label (with skull), seg_label, synthetic image, real image]
    h, w, c = np.array(syn_img).shape
    new_img = np.zeros((h, 4*w, c), dtype=np.uint8)
    new_img[:, :w, :] = np.array(label)
    new_img[:, w:2*w, :] = np.array(seg_label)
    new_img[:, 2*w:3*w, :] = np.array(syn_img)
    new_img[:, 3*w:, :] = np.array(real_img)
    new_img = Image.fromarray(new_img)

    new_img.save('{:s}/img_D{:d}_{:s}_{:d}_{:d}.png'.format(save_dir, dataset_num, img_path[0], index, append_idx))


def save_nuclei_data(gt, visuals, index, model, file, label_ternary, weight_map, dataset_num, append_idx, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    real_img = visuals['real_B'][index]
    syn_img = visuals['fake_B'][index]
    label = visuals['real_A'][index]
    label_ternary = label_ternary[index]
    weight_map = weight_map[index]

    img_path = model.get_image_paths()  # get image paths

    syn_img = syn_img.cpu().numpy()
    syn_img = (syn_img + 1) * (255 / 2)
    syn_img = syn_img.astype("uint8")
    syn_img = np.moveaxis(syn_img, 0, -1)

    label = label.cpu().numpy()
    label = (label + 1) * (255 / 2)
    label = label.astype("uint8")
    label = np.moveaxis(label, 0, -1)

    real_img = real_img.cpu().numpy()
    real_img = (real_img + 1) * (255 / 2)
    real_img = real_img.astype("uint8")
    real_img = np.moveaxis(real_img, 0, -1)

    # write to the h5 file
    file.create_dataset(f"images/D{dataset_num}_{img_path[0]}_{index}_{append_idx}", data=syn_img)
    file.create_dataset(f"labels/D{dataset_num}_{img_path[0]}_{index}_{append_idx}", data=label)
    file.create_dataset(f"real_images/D{dataset_num}_{img_path[0]}_{index}_{append_idx}", data=real_img)
    file.create_dataset(f"label_ternary/D{dataset_num}_{img_path[0]}_{index}_{append_idx}", data=label_ternary)
    file.create_dataset(f"weight_map/D{dataset_num}_{img_path[0]}_{index}_{append_idx}", data=weight_map.cpu().numpy())

    # save as png image file for better visualization: [label, synthetic image, real image]
    h, w, c = syn_img.shape
    new_img = np.zeros((h, 3*w, c), dtype=np.uint8)
    new_img[:, :w, :] = label
    new_img[:, w:2*w, :] = syn_img
    new_img[:, 2*w:, :] = real_img
    io.imsave('{:s}/img_D{:d}_{:s}_{:d}_{:d}.png'.format(save_dir, dataset_num, img_path[0], index, append_idx), new_img)


def launch_brain_test_once(dataset_num, idx, model, file, seg_label, save_dir):
    # test again.
    model.test()  # run inference
    visuals = model.get_current_visuals()  # get image results
    #
    gt = visuals['real_A']

    for j in range(gt.shape[0]):
        sys.stdout.flush()
        save_brain_data(gt, visuals, j, model, file, dataset_num, seg_label, idx, save_dir)


def launch_nuclei_test_once(dataset_num, idx, model, file, label_ternary, weight_map, save_dir):
    # test again.
    model.test()  # run inference
    visuals = model.get_current_visuals()  # get image results
    #
    gt = visuals['real_A']

    for j in range(gt.shape[0]):
        sys.stdout.flush()
        save_nuclei_data(gt, visuals, j, model, file, label_ternary, weight_map, dataset_num, idx, save_dir)


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0  # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))  # define the website directory
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()


    filename = f"{opt.name}_epoch{opt.epoch}_{opt.dataset_mode}"
    file = h5py.File(os.path.join(web_dir, f"{filename}.h5"), 'w')

    k = opt.task_num - 1

    if k == 0:    # nuclei
        for i, data in enumerate(dataset):
            print('Processing {:d}/{:d}'.format(i + 1, len(dataset)))
            model.set_input(data)  # unpack data from data loader
            label_ternary = data['label_ternary']
            weight_map = data['weight_map']
            launch_nuclei_test_once(k, 0, model, file, label_ternary, weight_map, '{:s}/images_{:s}'.format(web_dir, opt.dataset_mode))
    else:
        for i, data in enumerate(dataset):
            print('Processing {:d}/{:d}'.format(i + 1, len(dataset)))
            model.set_input(data)  # unpack data from data loader
            seg_label = data['Seg_label']
            launch_brain_test_once(k, 0, model, file, seg_label, '{:s}/images_{:s}'.format(web_dir, opt.dataset_mode))

    file.close()


