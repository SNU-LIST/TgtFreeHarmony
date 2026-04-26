"""
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import print_function
from utils import prepare_sub_folder, get_config, calculate_metric, get_data_loader, f1_score, f1_score_multi, setting_logger, cmap_convert
from trainer import Generator_Trainer
import argparse
from torch.autograd import Variable
import torchvision.utils as vutils
import sys
import torch
import os
from torchvision import transforms
from PIL import Image
import pickle
from data import pickle_loader
import pdb
from networks import UNet
import random
import logging_helper as logging_helper
import numpy as np
import torch.nn.functional as F
import cv2

from scipy.io import savemat        

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_num', type=str, default='0')

parser.add_argument('--output_folder', type=str, default='checkpoints/', help="output image path")
parser.add_argument('--BO_checkpoint', type=str, default='BboxHarmony/checkpoints/bo_best.pt', help="checkpoint of harmonization model") 

parser.add_argument('--pretrain_path', type=str, default='checkpoints/BboxHarmony/', help="pre-trained model path") 
parser.add_argument('--pretrain_file', type=str, default='loss', help="pre-trained model file name") 

parser.add_argument('--bbox_folder', type=str, default='./unet_seg/', help="output image path") 
parser.add_argument('--bbox_checkpoint', type=str, default='epoch_best.pth', help="checkpoint of disentangle model") 
parser.add_argument('--n_classes', type=int, default='4', help="number of classes for black-box segmentation model")

parser.add_argument('--style', type=str, default='', help="style image path")
parser.add_argument('--seed', type=int, default=10, help="random seed")
parser.add_argument('--output_path', type=str, default='.', help="path for logs, checkpoints, and VGG model weight")

parser.add_argument("--mask_toggle", type=bool, default=True, help="toggle for masking")
parser.add_argument("--save_toggle", type=bool, default=False, help="toggle for saving outputs")
parser.add_argument('--normalize_type', type=str, default='percentile', help="normalization type (percentile|minmax)")

parser.add_argument('--yaml_path', type=str, default='configs/bboxharmony_bo.yaml', help="path for yaml") 

opts = parser.parse_args()


torch.backends.cudnn.benchmark = True

torch.manual_seed(opts.seed)
torch.cuda.manual_seed(opts.seed)
if not os.path.exists(opts.output_folder):
    os.makedirs(opts.output_folder)

'''
Load experiment setting
'''
config_path = opts.yaml_path 
config = get_config(config_path)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= opts.gpu_num
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config['vgg_model_path'] = opts.output_path
style_dim = config['gen']['style_dim']

'''
logger setting
'''
logger = setting_logger(path=opts.output_folder, file_name='test_log.txt', module_type='test')

for key, value in vars(opts).items():
    logger.info('{:15s}: {}'.format(key,value))

'''
Setup dataloader
'''
test_loader = get_data_loader(logger, config, disentangle=False, zo_opt=True, phase='test', normalize=opts.normalize_type)

'''
Setup disentangle model
'''
pre_trainer = Generator_Trainer(config)
pre_trainer.to(device)
pretrain_directory = os.path.join(opts.pretrain_path)
pretrain_checkpoint_directory, _ = prepare_sub_folder(pretrain_directory)
pre_trainer.reload_genonly(pretrain_checkpoint_directory, model_name=opts.pretrain_file)
logger.info('*** Pre-trained disentangle model is loaded ***')
logger.info('Pre-trained disentangle model dir:' + str(pretrain_checkpoint_directory))

for param in pre_trainer.gen.parameters():
    param.requires_grad = False
pre_trainer.to(device)
pre_trainer.eval()
encode = pre_trainer.gen.encode
decode = pre_trainer.gen.decode

'''
Setup s_sample vector
'''
BO_ckpt_path = opts.output_folder + opts.BO_checkpoint
s_best_BO = torch.load(BO_ckpt_path)
s_best_BO = s_best_BO.to(device)

'''
Setup bbox downstream task model
'''
bbox_model = UNet(n_channels=1, n_classes=opts.n_classes, bilinear=True)
bbox_ckpt_path = opts.bbox_folder + opts.bbox_checkpoint
state_dict = torch.load(bbox_ckpt_path)
bbox_model.load_state_dict(state_dict)
bbox_model.to(device)
bbox_model.eval()

'''
Caculation of initial downstream-task metric without harmonization
'''
init_iou = 0
init_f1 = 0
init_psnr = 0
init_ssim = 0

bbox_inits = []

f1_init_list = []
f1_out_list = []

iou_list_mat, f1_list_mat = [], []
psnr_list_mat, ssim_list_mat = [], []

with torch.no_grad():
    bbox_model.eval()
    
    for images_a, images_b, mask, bbox_label in test_loader:           
        images_a, images_b, mask, bbox_label = images_a.to(device).detach(), images_b.to(device).detach(), mask.to(device).detach(), bbox_label.to(device).detach()

        if opts.mask_toggle is True:
            images_a = images_a * mask
            images_b = images_b * mask
        bbox_label = bbox_label * mask

        bbox_output = bbox_model(images_a)
        bbox_output = (bbox_output > 0)

        bbox_output = bbox_output * mask
        bbox_output = bbox_output.float()
        bbox_label = bbox_label.float()

        bbox_inits.append(bbox_output.data * mask)

        if opts.n_classes == 1:
            init_iou += torch.sum(bbox_output*bbox_label)/(torch.sum(bbox_output)+torch.sum(bbox_label)-torch.sum(bbox_output*bbox_label) + 1e-6)
            init_f1 += f1_score(bbox_output, bbox_label)
        else:
            labels_onehot = F.one_hot(bbox_label.squeeze(1).to(torch.int64), opts.n_classes).permute(0, 3, 1, 2)
            cur_iou = 0
            for ch in range(1, bbox_output.shape[1]):
                true_mask = (labels_onehot[:, ch] == 1).float()
                pred_mask = (bbox_output[:, ch] == 1)

                intersection = torch.sum(true_mask * pred_mask)  # Sum of pixel-wise intersection
                union = torch.sum(true_mask) + torch.sum(pred_mask) - intersection  # Sum of pixel-wise union

                # Calculate IoU (avoid division by zero)
                iou = intersection / (union + 1e-6)  # Add a small epsilon to avoid division by zero
                cur_iou += iou
            init_iou += (cur_iou/(bbox_output.shape[1]-1))

            iou_list_mat.append((cur_iou/(bbox_output.shape[1]-1)).item())

            cur_f1 = f1_score_multi(bbox_output, bbox_label.squeeze(1), num_classes=opts.n_classes) 
            init_f1 += (cur_f1[1:]).mean()

            f1_list_mat.append((cur_f1[1:]).mean().item())

            f1_init_list.append((cur_f1[1:]).mean())



        cur_psnr, cur_ssim = calculate_metric(images_a, images_b, mask)
        psnr_list_mat.append(float(cur_psnr))
        ssim_list_mat.append(float(cur_ssim))
        init_psnr += cur_psnr
        init_ssim += cur_ssim

init_iou /= len(test_loader.dataset)
init_f1 /= len(test_loader.dataset)
init_psnr /= len(test_loader.dataset)
init_ssim /= len(test_loader.dataset)

logger.info('* Init results *')
logger.info('Init) IoU: %.4f / f1: %.4f' % (init_iou, init_f1))
logger.info('Init) pSNR: %.4f / SSIM: %.4f' % (init_psnr, init_ssim))


save_path_mat = os.path.join(opts.output_folder, 'init_metrics_per_slice.mat')
savemat(save_path_mat, {
    'iou' : np.array(iou_list_mat,  dtype=np.float32),
    'f1'  : np.array(f1_list_mat,   dtype=np.float32),
    'psnr': np.array(psnr_list_mat, dtype=np.float32),
    'ssim': np.array(ssim_list_mat, dtype=np.float32)
})


with torch.no_grad():
    images = []
    masks = []

    content_input = []
    targets = []
    outputs = []
    bbox_outputs = []
    bbox_labels = []

    psnr_list = []
    f1_list = []

    output_pkl = []

    total_psnr = 0
    total_ssim = 0
    total_iou = 0
    total_f1 = 0

    for images_a, images_b, mask, bbox_label in test_loader:
        pre_trainer.eval()
        bbox_model.eval()

        images_a, images_b, mask, bbox_label = images_a.to(device).detach(), images_b.to(device).detach(), mask.to(device).detach(), bbox_label.to(device).detach()
        if opts.mask_toggle is True:
            images_a = images_a * mask
            images_b = images_b * mask
        bbox_label = bbox_label * mask

        encode = pre_trainer.gen.encode
        decode = pre_trainer.gen.decode

        content, style = encode(images_a)
        output = decode(content, s_best_BO) 

        if opts.mask_toggle is True:
            output = output * mask

        bbox_output = bbox_model(output)
        bbox_output = (bbox_output > 0)


        bbox_output = bbox_output * mask
        bbox_output = bbox_output.float()
        bbox_label = bbox_label.float()

        if opts.n_classes == 1:
            total_iou += torch.sum(bbox_output*bbox_label)/(torch.sum(bbox_output)+torch.sum(bbox_label)-torch.sum(bbox_output*bbox_label)+ 1e-6)
            total_f1 += f1_score(bbox_output, bbox_label)
        else:
            labels_onehot = F.one_hot(bbox_label.squeeze(1).to(torch.int64), opts.n_classes).permute(0, 3, 1, 2)
            cur_iou = 0
            for ch in range(1, bbox_output.shape[1]):
                true_mask = (labels_onehot[:, ch] == 1).float()
                pred_mask = (bbox_output[:, ch] == 1)

                intersection = torch.sum(true_mask * pred_mask)  # Sum of pixel-wise intersection
                union = torch.sum(true_mask) + torch.sum(pred_mask) - intersection  # Sum of pixel-wise union

                # Calculate IoU (avoid division by zero)
                iou = intersection / (union + 1e-6)  # Add a small epsilon to avoid division by zero
                cur_iou += iou
            total_iou += (cur_iou/(bbox_output.shape[1]-1))

            cur_f1 = f1_score_multi(bbox_output, bbox_label.squeeze(1), num_classes=opts.n_classes) 
            total_f1 += (cur_f1[1:]).mean()

            f1_out_list.append((cur_f1[1:]).mean())

        cur_psnr, cur_ssim = calculate_metric(output, images_b, mask)
        total_psnr += cur_psnr
        total_ssim += cur_ssim

        content_input.append(images_a.data * mask)
        targets.append(images_b.data * mask)
        outputs.append(output.data * mask)

        bbox_outputs.append(bbox_output.data * mask)
        bbox_labels.append(bbox_label.data * mask)

        psnr_list.append(cur_psnr)
        f1_list.append((cur_f1[1:]).mean())

logger.info('\n* Harmonization results*')
logger.info('Result) IoU: %.4f / f1: %.4f' % (total_iou/len(test_loader.dataset), total_f1/len(test_loader.dataset)))
logger.info('Result) pSNR: %.4f / SSIM: %.4f' % (total_psnr/len(test_loader.dataset), total_ssim/len(test_loader.dataset)))

if opts.save_toggle is True:
    if not os.path.exists(opts.output_folder + '/result'):
        os.makedirs(opts.output_folder + '/result')

    for idx_ in range(0, len(content_input)):
        img_name = str(round(psnr_list[idx_], 1)).replace('.', '_') + '_' + str(f1_list[idx_]).split('.')[1][:4] + '.png'

        path = os.path.join(opts.output_folder + '/result/', 'img_' + str(idx_) + '_1_src_' + img_name)
        vutils.save_image(torch.flip(content_input[idx_].permute(0,1,3,2), dims=(2,)),
                          path)
        
        path = os.path.join(opts.output_folder + '/result/', 'img_' + str(idx_) + '_img2_harOut_' + img_name)
        vutils.save_image(torch.flip(outputs[idx_].permute(0,1,3,2), dims=(2,)),
                          path)
        
        path = os.path.join(opts.output_folder + '/result/', 'img_' + str(idx_) + '_img3_trg_' + img_name)
        vutils.save_image(torch.flip(targets[idx_].permute(0,1,3,2), dims=(2,)),
                          path)

        bbox_init = torch.flip(bbox_inits[idx_].permute(0,1,3,2), dims=(2,))
        path = os.path.join(opts.output_folder + '/result/', 'img_' + str(idx_) + '_img4_segInit_' + img_name)
        if opts.n_classes > 1:
            cmap = cmap_convert()

            batch_size, _, height, width = bbox_init.shape
            colored_mask = np.zeros((height, width, 3), dtype=np.uint8)

            single_mask = bbox_init[0,:,:].cpu().numpy()
            for class_id, color in cmap.items():
                colored_mask[single_mask[class_id, :] == 1] = color

            cv2.imwrite(path, colored_mask)
        else:
            vutils.save_image(bbox_init,
                              path)
        
        bbox_output = torch.flip(bbox_outputs[idx_].permute(0,1,3,2), dims=(2,))
        path = os.path.join(opts.output_folder + '/result/', 'img_' + str(idx_) + '_img5_segOut_' + img_name)
        if opts.n_classes > 1:
            cmap = cmap_convert()

            batch_size, _, height, width = bbox_output.shape
            colored_mask = np.zeros((height, width, 3), dtype=np.uint8)

            single_mask = bbox_output[0,:,:].cpu().numpy()
            for class_id, color in cmap.items():
                colored_mask[single_mask[class_id, :] == 1] = color

            cv2.imwrite(path, colored_mask)
        else:
            vutils.save_image(bbox_output,
                              path)
        
        bbox_label = torch.flip(bbox_labels[idx_].permute(0,1,3,2), dims=(2,))
        path = os.path.join(opts.output_folder + '/result/', 'img_' + str(idx_) + '_img6_segLbl_' + img_name)
        if opts.n_classes > 1:
            cmap = cmap_convert()

            batch_size, _, height, width = bbox_label.shape
            colored_mask = np.zeros((height, width, 3), dtype=np.uint8)

            single_mask = bbox_label[0,0,:,:].cpu().numpy()
            for class_id, color in cmap.items():
                colored_mask[(single_mask == class_id), :] = color

            cv2.imwrite(path, colored_mask)
        else:
            vutils.save_image(bbox_label,
                              path)
else:
    random_idx = [50, 180, 250, 280, 320, 400, 480, 520]

    # content
    content_grid = [content_input[i] for i in random_idx]
    content_grid = torch.cat(content_grid, dim=0)
    content_grid = content_grid.permute(0,1,3,2)
    content_grid = torch.flip(content_grid, dims=(2,))
    path = os.path.join(opts.output_folder, 'img1_source.jpg')
    vutils.save_image(content_grid, path, nrow=len(random_idx))

    # target
    style_grid = [targets[i] for i in random_idx]
    style_grid = torch.cat(style_grid, dim=0)
    style_grid = style_grid.permute(0,1,3,2)
    style_grid = torch.flip(style_grid, dims=(2,))
    path = os.path.join(opts.output_folder, 'img2_target.jpg')
    vutils.save_image(style_grid, path, nrow=len(random_idx))

    # harmonizer output
    output_grid = [outputs[i] for i in random_idx]
    output_grid = torch.cat(output_grid, dim=0)
    output_grid = output_grid.permute(0,1,3,2)
    output_grid = torch.flip(output_grid, dims=(2,))
    path = os.path.join(opts.output_folder, 'img3_harmonizer_output.jpg')
    vutils.save_image(output_grid, path, nrow=len(random_idx))

    # bbox label
    bbox_label_grid = [bbox_labels[i] for i in random_idx]
    bbox_label_grid = torch.cat(bbox_label_grid, dim=0)
    bbox_label_grid = bbox_label_grid.permute(0,1,3,2)
    bbox_label_grid = torch.flip(bbox_label_grid, dims=(2,))

    if opts.n_classes > 1:
        batch_size, _, height, width = bbox_label_grid.shape

        n_col = len(random_idx)
        n_row = (batch_size / n_col)

        grid_height = height * n_row
        grid_width = width * n_col

        grid = bbox_label_grid.view(batch_size, height, width).detach().cpu().numpy()

        colored_image = np.zeros((int(grid_height), int(grid_width), 3), dtype=np.uint8)
        color_map = {
            0: (0, 0, 0),  # Black
            1: (255, 0, 0),  # Red
            2: (0, 255, 0),  # Green
            3: (0, 0, 255),  # Blue
        }

        for idx, mask in enumerate(grid):
            row_idx = idx // n_col
            col_idx = idx % n_col
            y_start = row_idx * height
            y_end = y_start + height
            x_start = col_idx * width
            x_end = x_start + width

            for class_value, color in color_map.items():
                mask_region = (mask == class_value)
                colored_image[y_start:y_end, x_start:x_end, :][mask_region] = color

        image = Image.fromarray(colored_image)
        path = os.path.join(opts.output_folder, 'img4_harmonizer_label_colored.png')
        image.save(path)
    else:
        path = os.path.join(opts.output_folder, 'img4_harmonizer_label.jpg')
        vutils.save_image(bbox_label_grid, path, nrow=len(test_loader.dataset))


    # bbox output
    bbox_output_grid = [bbox_outputs[i] for i in random_idx]

    if opts.n_classes > 1:
        bbox_output_total = []
        for i in range(0, len(bbox_output_grid)):
            temp_bbox_output = bbox_output_grid[i]
            temp_total = torch.zeros((1, temp_bbox_output.shape[2], temp_bbox_output.shape[3]), dtype=torch.long)
            for ch in range(1, 4):
                temp_total[temp_bbox_output[:, ch, :, :] > 0] = ch
            bbox_output_total.append(temp_total.unsqueeze(0))

        bbox_output_grid = bbox_output_total
        bbox_output_grid = torch.cat(bbox_output_grid, dim=0)
        bbox_output_grid = bbox_output_grid.permute(0,1,3,2)
        bbox_output_grid = torch.flip(bbox_output_grid, dims=(2,))

        batch_size, _, height, width = bbox_output_grid.shape

        n_col = len(random_idx)
        n_row = (batch_size / n_col)

        grid_height = height * n_row
        grid_width = width * n_col

        grid = bbox_output_grid.view(batch_size, height, width).detach().cpu().numpy()

        colored_image = np.zeros((int(grid_height), int(grid_width), 3), dtype=np.uint8)
        color_map = {
            0: (0, 0, 0),  # Black
            1: (255, 0, 0),  # Red
            2: (0, 255, 0),  # Green
            3: (0, 0, 255),  # Blue
        }

        for idx, mask in enumerate(grid):
            row_idx = idx // n_col
            col_idx = idx % n_col
            y_start = row_idx * height
            y_end = y_start + height
            x_start = col_idx * width
            x_end = x_start + width

            for class_value, color in color_map.items():
                mask_region = (mask == class_value)
                colored_image[y_start:y_end, x_start:x_end, :][mask_region] = color

        image = Image.fromarray(colored_image)
        path = os.path.join(opts.output_folder, 'img5_harmonizer_output_colored.png')
        image.save(path)
    else:
        path = os.path.join(opts.output_folder, 'img5_harmonizer_output.jpg')
        vutils.save_image(bbox_output_grid, path, nrow=len(test_loader.dataset))

    # bbox output (no harmonized)
    bbox_init_grid = [bbox_inits[i] for i in random_idx]

    if opts.n_classes > 1:
        bbox_init_total = []
        for i in range(0, len(bbox_init_grid)):
            temp_bbox_init = bbox_init_grid[i]
            temp_total = torch.zeros((1, temp_bbox_init.shape[2], temp_bbox_init.shape[3]), dtype=torch.long)
            for ch in range(1, 4):
                temp_total[temp_bbox_init[:, ch, :, :] > 0] = ch
            bbox_init_total.append(temp_total.unsqueeze(0))

        bbox_init_grid = bbox_init_total
        bbox_init_grid = torch.cat(bbox_init_grid, dim=0)
        bbox_init_grid = bbox_init_grid.permute(0,1,3,2)
        bbox_init_grid = torch.flip(bbox_init_grid, dims=(2,))

        batch_size, _, height, width = bbox_init_grid.shape

        n_col = len(random_idx)
        n_row = (batch_size / n_col)

        grid_height = height * n_row
        grid_width = width * n_col

        grid = bbox_init_grid.view(batch_size, height, width).detach().cpu().numpy()

        colored_image = np.zeros((int(grid_height), int(grid_width), 3), dtype=np.uint8)
        color_map = {
            0: (0, 0, 0),  # Black
            1: (255, 0, 0),  # Red
            2: (0, 255, 0),  # Green
            3: (0, 0, 255),  # Blue
        }

        for idx, mask in enumerate(grid):
            row_idx = idx // n_col
            col_idx = idx % n_col
            y_start = row_idx * height
            y_end = y_start + height
            x_start = col_idx * width
            x_end = x_start + width

            for class_value, color in color_map.items():
                mask_region = (mask == class_value)
                colored_image[y_start:y_end, x_start:x_end, :][mask_region] = color

        image = Image.fromarray(colored_image)
        path = os.path.join(opts.output_folder, 'img6_bbox_init_colored.png')
        image.save(path)
    else:
        path = os.path.join(opts.output_folder, 'img6_bbox_init.jpg')
        vutils.save_image(bbox_init_grid, path, nrow=len(test_loader.dataset))