"""
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch.utils.data as data
import os.path
import pickle
import cv2
import time
import datetime
import random
import numpy as np
import pdb
import torchvision.utils as vutils
import torch
import matplotlib.pyplot as plt


# percentile + minmax total
def pickle_loader(img, normalize='percentile'):
    img = img.astype('float32')

    if normalize == 'percentile':
        if np.max(img) > 1:
            lower_percentile = 1
            upper_percentile = 99

            min_val = np.percentile(img, lower_percentile)
            max_val = np.percentile(img, upper_percentile)
            img = np.clip(img, min_val, max_val)
            img = (img - min_val) / (max_val - min_val)
            img = np.clip(img, 0, 1)
    elif normalize == 'minmax':
        if np.max(img) > 1:
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
            img = np.clip(img, 0, 1)
    return img

def default_pickle_loader(img):
    img = img.astype('float32')
    return img

def default_flist_reader(flist):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath = line.strip()
            imlist.append(impath)

    return imlist



import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


def contrast_adjustment(img, alpha_value, beta_value):
    # Ensure img is in range [0, 255] for cv2 processing
    img_aug = img.squeeze()
    img_aug = img * 255.0 if np.max(img) <= 1.0 else img

    img_aug = cv2.convertScaleAbs(img_aug, alpha=alpha_value, beta=beta_value)

    # Normalize back to [0, 1]
    img_aug = img_aug / 255.0
    img_aug = np.float32(img_aug)
    return img_aug

def gamma_transform(img, power_value):
    img = img.squeeze()
    img_aug = np.power(img, power_value)

    # Normalize back to [0, 1]
    img_aug = (img_aug - np.min(img_aug)) / (np.max(img_aug) - np.min(img_aug)) if np.max(img_aug) > 1 else img_aug
    return img_aug

def gaussian_blur(img, kernel_size, sigma_value):
    img = img.squeeze()
    img_aug = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma_value)
    
    # Normalize back to [0, 1]
    img_aug = (img_aug - np.min(img_aug)) / (np.max(img_aug) - np.min(img_aug)) if np.max(img_aug) > 1 else img_aug
    return img_aug

def gaussian_noise(img, mu, sigma):
    img = img.squeeze()
    random_noise = np.random.normal(mu, sigma, img.shape)
    img_aug = img + random_noise

    # Normalize back to [0, 1]
    img_aug = (img_aug - np.min(img_aug)) / (np.max(img_aug) - np.min(img_aug)) if np.max(img_aug) > 1 else img_aug
    return img_aug

    
class ImagePickle_one_encdec(data.Dataset):
    def __init__(self, logger, conf, phase, normalize, transform=None, loader=pickle_loader):
        file_list = conf['sub_folders']

        if phase == 'train':
            file_path1 = conf['data_root'] + 'vendor0/pkl2/vendor0_train_pkl2.pklv4'
            mask_path1 = conf['data_root'] + 'vendor0/pkl2/vendor0_seg_train_pkl2.pklv4'

            file_path2 = conf['data_root'] + 'vendor21926/pkl2/vendor21926_train_pkl2.pklv4'
            mask_path2 = conf['data_root'] + 'vendor21926/pkl2/vendor21926_seg_train_pkl2.pklv4'

            file_path3 = conf['data_root'] + 'vendor175614/pkl2/vendor175614_train_pkl2.pklv4'
            mask_path3 = conf['data_root'] + 'vendor175614/pkl2/vendor175614_seg_train_pkl2.pklv4'
            

        elif phase == 'val':
            file_path1 = conf['data_root'] + 'vendor0/pkl2/vendor0_val_pkl2.pklv4'
            mask_path1 = conf['data_root'] + 'vendor0/pkl2/vendor0_seg_val_pkl2.pklv4'

            file_path2 = conf['data_root'] + 'vendor21926/pkl2/vendor21926_val_pkl2.pklv4'
            mask_path2 = conf['data_root'] + 'vendor21926/pkl2/vendor21926_seg_val_pkl2.pklv4'

            file_path3 = conf['data_root'] + 'vendor175614/pkl2/vendor175614_val_pkl2.pklv4'
            mask_path3 = conf['data_root'] + 'vendor175614/pkl2/vendor175614_seg_val_pkl2.pklv4'
            
        elif phase == 'test':
            file_path = conf['data_root'] + 'vendor' + file_list[0] + '/pkl2/vendor' + file_list[0] + '_test_pkl2.pklv4'
            mask_path = conf['data_root'] + 'vendor' + file_list[0] + '/pkl2/vendor' + file_list[0] + '_seg_test_pkl2.pklv4'
        else:
            raise(RuntimeError("Check the parameters: " + phase + "\n" + "Supported parameters are: train val test"))

        logger.info(str(phase) + ' data:' + str(file_path1) + '\n' + str(file_path2) + '\n'+ str(file_path3))

        imgs = []
        msks = []

        assert os.path.isfile(file_path1), file_path1
        assert os.path.isfile(mask_path1), mask_path1
        assert os.path.isfile(file_path2), file_path2
        assert os.path.isfile(mask_path2), mask_path2
        assert os.path.isfile(file_path3), file_path3
        assert os.path.isfile(mask_path3), mask_path3

        with open(file_path1, "rb") as f:
            imgs += pickle.load(f)
        with open(file_path2, "rb") as f:
            imgs += pickle.load(f)
        with open(file_path3, "rb") as f:
            imgs += pickle.load(f)

        with open(mask_path1, "rb") as f:
            msks += pickle.load(f)
        with open(mask_path2, "rb") as f:
            msks += pickle.load(f)
        with open(mask_path3, "rb") as f:
            msks += pickle.load(f)

        assert len(imgs) == len(msks), 'Check the img-msk pairs !!!'

        logger.info('data len: ' + str(len(imgs)))
        
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + file_path + "\n" + "Supported image extension is: pklv4"))
        elif len(msks) == 0:
            raise(RuntimeError("Found 0 images in: " + mask_path + "\n" + "Supported image extension is: pklv4"))

        self.imgs = imgs
        self.msks = msks
        self.normalize = normalize
        self.transform = transform
        self.loader = loader
        self.aug_contrast = conf['aug_contrast']
        self.aug_gamma = conf['aug_gamma']
        self.aug_blur = conf['aug_blur']
        self.aug_noise = conf['aug_noise']
        self.aug_toggle = self.aug_contrast or self.aug_gamma or self.aug_blur or self.aug_noise
        
        self.aug_contrast_alpha_min = conf['aug_contrast_alpha_min']
        self.aug_contrast_alpha_max = conf['aug_contrast_alpha_max']
        self.aug_contrast_beta_min = conf['aug_contrast_beta_min']
        self.aug_contrast_beta_max = conf['aug_contrast_beta_max']

        self.aug_gamma_min = conf['aug_gamma_min']
        self.aug_gamma_max = conf['aug_gamma_max']

        self.aug_blur_kernel_size = conf['aug_blur_kernel_size']
        self.aug_blur_min = conf['aug_blur_sigma_min']
        self.aug_blur_max = conf['aug_blur_sigma_max']

        self.aug_noise_mu = conf['aug_noise_mu']
        self.aug_noise_sigma_min = conf['aug_noise_sigma_min']
        self.aug_noise_sigma_max = conf['aug_noise_sigma_max']

    def __getitem__(self, index):
        img = self.loader(self.imgs[index], normalize=self.normalize)
        msk = self.loader(self.msks[index], normalize=None)

        msk = (msk != 0)

        if self.aug_toggle is True:
            img_aug = img

            '''
            random perturbation
            '''
            np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)

            alpha_value = random.uniform(self.aug_contrast_alpha_min, self.aug_contrast_alpha_max)
            beta_value = random.uniform(self.aug_contrast_beta_min, self.aug_contrast_beta_max)

            power_value = random.uniform(self.aug_gamma_min, self.aug_gamma_max)

            kernel_size = self.aug_blur_kernel_size
            blur_sigma_value = random.uniform(self.aug_blur_min, self.aug_blur_max)

            noise_sigma_value = random.uniform(self.aug_noise_sigma_min, self.aug_noise_sigma_max)

            '''
            Fixed perturbation
            '''
            
            # random contrast adjustment
            img_aug = contrast_adjustment(img_aug, alpha_value, beta_value) if self.aug_contrast is True else img_aug

            # random gamma transform
            img_aug = gamma_transform(img_aug, power_value) if self.aug_gamma is True else img_aug

            # random gaussian blur
            img_aug = gaussian_blur(img_aug, kernel_size, blur_sigma_value) if self.aug_blur is True else img_aug

            # random gaussign noise
            img_aug = gaussian_noise(img_aug, self.aug_noise_mu, noise_sigma_value) if self.aug_noise is True else img_aug
            
            img_aug = np.expand_dims(img_aug, axis=2)
            img_aug = np.float32(img_aug)
        else:
            img_aug = img # dummy data

        if self.transform is not None:
            img = self.transform(img)
            img_aug = self.transform(img_aug)
            msk = self.transform(msk)

        return img, img_aug, msk

    def __len__(self):
        return len(self.imgs)


class ImagePickle_pseudo_trg(data.Dataset):
    def __init__(self, logger, conf, zo_opt, phase, normalize, transform=None, loader=pickle_loader):
        file_list = conf['sub_folders']

        logger.info('*** traveling dataset was used ***')
        if phase == 'train':
            if zo_opt is False:
                src_path = conf['data_root'] + 'vendor' + file_list[0] + '/pkl2/vendor' + file_list[0] + '_ori_train_perturbed4_to_35177_10_multi_nomask_munit_pkl2.pklv4'
                trg_path = conf['data_root'] + 'vendor' + file_list[0] + '/pkl2/vendor' + file_list[0] + '_train_perturbed4_to_35177_10_multi_nomask_munit_pkl2.pklv4'
                mask_path = conf['data_root'] + 'vendor' + file_list[0] + '/pkl2/vendor' + file_list[0] + '_seg_train_perturbed4_to_35177_10_multi_nomask_munit_pkl2.pklv4'
            else:
                print('zoo opt + step1 ver.')
                if file_list[0] == '175614':
                    src_path = conf['data_root'] + file_list[0] + '_35177_paired_acc/pkl2/vendor' + file_list[0] + '_35177pair_' + file_list[0] + '_train_5_pkl2.pklv4'
                    trg_path = conf['data_root'] + file_list[0] + '_35177_paired_acc/pkl2/vendor' + file_list[0] + '_35177pair_' + '35177' + '_train_5_pkl2.pklv4'
                    mask_path = conf['data_root'] + file_list[0] + '_35177_paired_acc/pkl2/vendor' + file_list[0] + '_35177pair_' + file_list[0] + '_train_seg_5_pkl2.pklv4'
                else:
                    src_path = conf['data_root'] + file_list[0] + '_35177_paired/pkl2/vendor' + file_list[0] + '_35177pair_' + file_list[0] + '_train_5_pkl2.pklv4'
                    trg_path = conf['data_root'] + file_list[0] + '_35177_paired/pkl2/vendor' + file_list[0] + '_35177pair_' + '35177' + '_train_5_pkl2.pklv4'
                    mask_path = conf['data_root'] + file_list[0] + '_35177_paired/pkl2/vendor' + file_list[0] + '_35177pair_' + file_list[0] + '_train_seg_5_pkl2.pklv4'
            
        elif phase == 'val':
            if zo_opt is False:
                src_path = conf['data_root'] + 'vendor' + file_list[0] + '/pkl2/vendor' + file_list[0] + '_ori_val_perturbed4_to_35177_10_multi_nomask_munit_pkl2.pklv4'
                trg_path = conf['data_root'] + 'vendor' + file_list[0] + '/pkl2/vendor' + file_list[0] + '_val_perturbed4_to_35177_10_multi_nomask_munit_pkl2.pklv4'
                mask_path = conf['data_root'] + 'vendor' + file_list[0] + '/pkl2/vendor' + file_list[0] + '_seg_val_perturbed4_to_35177_10_multi_nomask_munit_pkl2.pklv4'
            else:
                
                print('zoo opt + step1 ver.')             
                if file_list[0] == '175614':
                    src_path = conf['data_root'] + file_list[0] + '_35177_paired_acc/pkl2/vendor' + file_list[0] + '_35177pair_' + file_list[0] + '_val_5_pkl2.pklv4'
                    trg_path = conf['data_root'] + file_list[0] + '_35177_paired_acc/pkl2/vendor' + file_list[0] + '_35177pair_' + '35177' + '_val_5_pkl2.pklv4'
                    mask_path = conf['data_root'] + file_list[0] + '_35177_paired_acc/pkl2/vendor' + file_list[0] + '_35177pair_' + file_list[0] + '_val_seg_5_pkl2.pklv4'
                else:
                    src_path = conf['data_root'] + file_list[0] + '_35177_paired/pkl2/vendor' + file_list[0] + '_35177pair_' + file_list[0] + '_val_5_pkl2.pklv4'
                    trg_path = conf['data_root'] + file_list[0] + '_35177_paired/pkl2/vendor' + file_list[0] + '_35177pair_' + '35177' + '_val_5_pkl2.pklv4'
                    mask_path = conf['data_root'] + file_list[0] + '_35177_paired/pkl2/vendor' + file_list[0] + '_35177pair_' + file_list[0] + '_val_seg_5_pkl2.pklv4'
            
        elif phase == 'test':
            if file_list[0] == '175614':
                src_path = conf['data_root'] + file_list[0] + '_35177_paired_acc/pkl2/vendor' + file_list[0] + '_35177pair_' + file_list[0] + '_test_5_pkl2.pklv4'
                trg_path = conf['data_root'] + file_list[0] + '_35177_paired_acc/pkl2/vendor' + file_list[0] + '_35177pair_' + '35177' + '_test_5_pkl2.pklv4'
                mask_path = conf['data_root'] + file_list[0] + '_35177_paired_acc/pkl2/vendor' + file_list[0] + '_35177pair_' + file_list[0] + '_test_seg_5_pkl2.pklv4'
            else:
                src_path = conf['data_root'] + file_list[0] + '_35177_paired/pkl2/vendor' + file_list[0] + '_35177pair_' + file_list[0] + '_test_5_pkl2.pklv4'
                trg_path = conf['data_root'] + file_list[0] + '_35177_paired/pkl2/vendor' + file_list[0] + '_35177pair_' + '35177' + '_test_5_pkl2.pklv4'
                mask_path = conf['data_root'] + file_list[0] + '_35177_paired/pkl2/vendor' + file_list[0] + '_35177pair_' + file_list[0] + '_test_seg_5_pkl2.pklv4' 
        else:
            raise(RuntimeError("Check the parameters: " + phase + "\n" + "Supported parameters are: train val test"))

        src_imgs = []
        trg_imgs = []
        msks = []
        
        assert os.path.isfile(src_path), src_path
        assert os.path.isfile(trg_path), trg_path
        assert os.path.isfile(mask_path), mask_path

        with open(src_path, "rb") as f:
            src_imgs += pickle.load(f)
        with open(trg_path, "rb") as f:
            trg_imgs += pickle.load(f)
        with open(mask_path, "rb") as f:
            msks += pickle.load(f)

        if conf['crop_use']:
            print("For Swin_Unet datas all crop to org size + Train!!!!!!!!!!!!!!")
            src_imgs = [img[40:-40, 24:-24,...] for img in src_imgs]
            trg_imgs = [img[40:-40, 24:-24,...] for img in trg_imgs]
            msks = [img[40:-40, 24:-24,...] for img in msks]

        assert len(src_imgs) == len(msks), 'Check the img-msk pairs !!!'
        assert len(src_imgs) == len(trg_imgs), 'Check the img pairs !!!'

        logger.info(str(phase) + ' src data:' + str(src_path))
        logger.info(str(phase) + ' trg data:' + str(trg_path))
        logger.info('data len: %d' %(len(src_imgs)))
        
        if len(src_imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + src_path + "\n" + "Supported image extension is: pklv4"))
        elif len(trg_imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + trg_path + "\n" + "Supported image extension is: pklv4"))
        elif len(msks) == 0:
            raise(RuntimeError("Found 0 images in: " + mask_path + "\n" + "Supported image extension is: pklv4"))

        print('src:', src_imgs[0].max())
        print('trg:', trg_imgs[0].max())

        self.src_imgs = src_imgs
        self.trg_imgs = trg_imgs
        self.msks = msks
        self.transform = transform
        self.normalize = normalize
        self.loader = loader
        self.loader_default = default_pickle_loader
        self.zo_opt = zo_opt

    def __getitem__(self, index):
        src_img = self.loader(self.src_imgs[index], normalize=self.normalize)
        trg_img = self.loader(self.trg_imgs[index], normalize=self.normalize)
        msk = self.loader(self.msks[index], normalize=None)

        if self.zo_opt is True:
            bbox_label = msk

        msk = (msk != 0)

        if self.transform is not None:
            src_img = self.transform(src_img)
            trg_img = self.transform(trg_img)
            msk = self.transform(msk)

            if self.zo_opt is True:
                bbox_label = self.transform(bbox_label)

        if self.zo_opt is True:
            return src_img, trg_img, msk, bbox_label.to(torch.long)
        else:
            return src_img, trg_img, msk

    def __len__(self):
        return len(self.src_imgs)
