"""
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from torch.optim import Adam

from utils import SSIM_cal_2D, get_data_loader, prepare_sub_folder, get_config, calculate_metric, setting_logger, f1_score, f1_score_multi
import argparse
from torch.autograd import Variable
from trainer import Generator_Trainer
import torch.backends.cudnn as cudnn
import torch
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
import os
import sys
import shutil
import pdb
import logging_helper as logging_helper
from tqdm import tqdm
from networks import UNet
import torchvision.utils as vutils
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
import gpytorch 

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_num', type=str, default='0')
parser.add_argument('--config', type=str, default='configs/bboxharmony_bo.yaml', help='Path to the config file.')

parser.add_argument('--pretrain_path', type=str, default='checkpoints/BboxHarmony/', help="pre-trained model path") 
parser.add_argument('--pretrain_file', type=str, default='loss', help="pre-trained model file name") 

parser.add_argument('--bbox_path', type=str, default='./unet_seg/', help="black-box model path")
parser.add_argument('--bbox_file', type=str, default='epoch_best.pth', help="black-box model file name")
parser.add_argument('--n_classes', type=int, default=4, help="number of classes for black-box model")

parser.add_argument('--output_path', type=str, default='checkpoints/temp_bo/', help="outputs path")
parser.add_argument("--mask_toggle", type=bool, default=True, help="toggle for masking")
parser.add_argument('--normalize_type', type=str, default='percentile', help="normalization type (percentile|minmax)")

parser.add_argument("--zoo_criterion", type=str, default='f1', help="f1|ssim") 

opts = parser.parse_args()

cudnn.benchmark = True


'''
Load experiment setting
'''
config = get_config(opts.config)
max_iter = config['max_iter']
display_size = config['display_size']
config['vgg_model_path'] = opts.output_path

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= opts.gpu_num   #"1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
logger setting
'''
logger = setting_logger(path=opts.output_path, file_name='train_log.txt', module_type='train')
writer = SummaryWriter(opts.output_path + 'runs/')

for key, value in vars(opts).items():
    logger.info('{:15s}: {}'.format(key,value))

model_name = os.path.splitext(os.path.basename(opts.config))[0]
logger.info('model name:' + str(model_name))
output_directory = os.path.join(opts.output_path)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml'))

'''
Setup dataloader
'''
train_loader_a = get_data_loader(logger, config, disentangle=False, zo_opt=True, phase='train', normalize=opts.normalize_type)
test_loader_a = get_data_loader(logger, config, disentangle=False, zo_opt=True, phase='val', normalize=opts.normalize_type)

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

''' 
Setup black-box model
'''
logger.info('*** Black-box model is loaded ***')
logger.info('Pre-trained bbox model dir:' + str(opts.bbox_path))
bbox_model = UNet(n_channels=1, n_classes=opts.n_classes, bilinear=True)
bbox_ckpt_path = opts.bbox_path + opts.bbox_file
state_dict = torch.load(bbox_ckpt_path)
bbox_model.load_state_dict(state_dict)
bbox_model.to(device)
bbox_model.eval()

'''
Setup Oracle metric
'''
ssim_cal_2d = SSIM_cal_2D()    

'''
Caculation of initial downstream-task metric without harmonization
'''
init_iou = 0
init_f1 = 0
init_psnr = 0
init_ssim = 0

with torch.no_grad():
    bbox_model.eval()
    
    for images_a, images_b, mask, bbox_label in test_loader_a:           
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
            init_iou += (cur_iou/bbox_output.shape[1])

            cur_f1 = f1_score_multi(bbox_output, bbox_label.squeeze(1), num_classes=opts.n_classes) 
            init_f1 += (cur_f1[1:]).mean()

        cur_psnr, cur_ssim = calculate_metric(images_a, images_b, mask)
        init_psnr += cur_psnr
        init_ssim += cur_ssim

init_iou /= len(test_loader_a.dataset)
init_f1 /= len(test_loader_a.dataset)
init_psnr /= len(test_loader_a.dataset)
init_ssim /= len(test_loader_a.dataset)

logger.info('Init metric of bbox model) IoU: %.4f / f1: %.4f' % (init_iou, init_f1))
logger.info('Init harmonization metric) pSNR: %.4f / SSIM: %.4f' % (init_psnr, init_ssim))


'''
Setup Blackbox function 
'''
def black_box_function(style_vector):
    style_vector = style_vector.view(1, -1, 1, 1).to(device)
    total_score = []
    
    with torch.no_grad():
        pre_trainer.gen.eval()
        bbox_model.eval()
        
        for images_a, images_b, mask, bbox_label in train_loader_a:
            images_a, images_b, mask, bbox_label = (
                images_a.to(device),
                images_b.to(device),
                mask.to(device),
                bbox_label.to(device),
            )

            if opts.mask_toggle:
                images_a = images_a * mask
                images_b = images_b * mask
                bbox_label = bbox_label * mask

            content, _ = pre_trainer.gen.encode(images_a)
            style_repeat = style_vector.expand(images_a.size(0), -1, -1, -1)
            outputs = pre_trainer.gen.decode(content, style_repeat)

            if opts.mask_toggle:
                outputs = outputs * mask

            bbox_output = bbox_model(outputs)
            bbox_output = (bbox_output > 0).float() * mask

            if opts.zoo_criterion == 'ssim':
                B = outputs.size(0)
                ones = torch.ones(B, device=device)
                score = ssim_cal_2d(outputs, images_b, ones).mean()
            elif opts.zoo_criterion == 'f1':
                score = f1_score_multi(bbox_output, bbox_label.squeeze(1), num_classes=opts.n_classes)[1:].mean() 
            else:
                raise ValueError("Unsupported criterion")
            
            total_score.append(score) 

    return torch.stack(total_score).mean()            

''' 
Setup Gaussian Process model
'''
# GP model definition
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=config['gen']['style_dim']) 
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Bayesian optimization initialization
style_dim = config['gen']['style_dim']
X_train_all = []
y_train_all = []

init_samples = torch.randn(100, style_dim, device=device) 
init_scores = []

print("init sample evaluation start!!")
best_init_score = -float('inf')
best_init_sample = None

# init eval --> best point explore
for idx, sample in enumerate(init_samples):
    score = black_box_function(sample)
    init_scores.append(score)
    print(f"[Init Sample {idx}] score={score:.4f}")

    if score > best_init_score:
        best_init_score = score
        best_init_sample = sample.clone()

print(f"Init Sample Best Score={best_init_score:.4f}\n")


init_scores_tensor = torch.tensor([s.item() for s in init_scores], device=device) 

# init samples accumulation
X_train_all.append(init_samples)
y_train_all.append(init_scores_tensor)

X_train = torch.cat(X_train_all).to(device)
y_train = torch.cat(y_train_all).to(device)

# GP model initialization
likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
model = ExactGPModel(X_train, y_train, likelihood).to(device)

# GP model initial training
model.train()
likelihood.train()
optimizer = Adam(model.parameters(), lr=0.1)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for epoch in range(50):
    optimizer.zero_grad()
    output = model(X_train)
    loss = -mll(output, y_train)
    loss.backward()
    optimizer.step()

# Bayesian optimization loop setting
bo_max_iter = 200 # org : 20
threshold = 1e-3
best_score = best_init_score
best_point = best_init_sample.clone()
iter_num_best = 0

for iter_num in range(bo_max_iter):
    model.eval()
    likelihood.eval()

    # candidate samples making
    candidates = torch.randn(100, style_dim, device=device) 

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        preds = model(candidates)
        mu = preds.mean
        sigma = preds.stddev.clamp_min(1e-9)

        # GP-UCB acq function
        beta = 0.1
        acq = mu + beta * sigma

    # next best point selection
    best_idx = acq.argmax()
    next_point = candidates[best_idx] 

    # black-box model perfomeance
    next_score = black_box_function(next_point)  

    # Data Accumulation
    X_train_all.append(next_point.unsqueeze(0)) 
    y_train_all.append(next_score.unsqueeze(0)) 
    X_train = torch.cat(X_train_all).to(device)
    y_train = torch.cat(y_train_all).to(device)

    # GP model update
    model.set_train_data(X_train, y_train, strict=False)
    model.train()
    likelihood.train()
    optimizer = Adam(model.parameters(), lr=0.1)

    for epoch in range(50):
        optimizer.zero_grad()
        output = model(X_train)
        loss = -mll(output, y_train)
        loss.backward()
        optimizer.step()

    # Improvement check
    improvement = next_score - best_score
    print(f"[Iter {iter_num:02d}]"
          f"score={next_score:.4f}, improvement={improvement:.5f}")
    if improvement > 0:
        best_score = next_score
        best_point = next_point.clone() #[32]
        best_point = best_point.unsqueeze(0).view(1, -1, 1, 1) #[1,32,1,1]
        iter_num_best = iter_num
    elif abs(improvement) < threshold:
        print("Early stopping triggered.")
        break
    #image_save
    next_point_dim4 = next_point.unsqueeze(0).view(1, -1, 1, 1) #[1,32,1,1]

    dbg_dir = os.path.join(image_directory, f'epoch_{iter_num:03d}')
    os.makedirs(dbg_dir, exist_ok=True)
    # 

    # s_sample_results
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

        f1_out_list = []

        output_pkl = []

        total_psnr = 0
        total_ssim = 0
        total_iou = 0
        total_f1 = 0

        for images_a, images_b, mask, bbox_label in test_loader_a:
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
            output = decode(content, next_point_dim4) 

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
    logger.info('Result) IoU: %.4f (init: %.4f) / f1: %.4f (init: %.4f)' % (total_iou/len(test_loader_a.dataset),init_iou, total_f1/len(test_loader_a.dataset),init_f1))
    logger.info('Result) pSNR: %.4f (init: %.4f) / SSIM: %.4f (init: %.4f) ' % (total_psnr/len(test_loader_a.dataset),init_psnr, total_ssim/len(test_loader_a.dataset),init_ssim))

    random_idx = [7,14,21,28,37]

    # content
    content_grid = [content_input[i] for i in random_idx]
    content_grid = torch.cat(content_grid, dim=0)
    content_grid = content_grid.permute(0,1,3,2)
    content_grid = torch.flip(content_grid, dims=(2,))
    path = os.path.join(dbg_dir, 'img1_source.jpg')
    vutils.save_image(content_grid, path, nrow=len(random_idx))

    # target
    style_grid = [targets[i] for i in random_idx]
    style_grid = torch.cat(style_grid, dim=0)
    style_grid = style_grid.permute(0,1,3,2)
    style_grid = torch.flip(style_grid, dims=(2,))
    path = os.path.join(dbg_dir, 'img2_target.jpg')
    vutils.save_image(style_grid, path, nrow=len(random_idx))

    # harmonizer output
    output_grid = [outputs[i] for i in random_idx]
    output_grid = torch.cat(output_grid, dim=0)
    output_grid = output_grid.permute(0,1,3,2)
    output_grid = torch.flip(output_grid, dims=(2,))
    path = os.path.join(dbg_dir, 'img3_harmonizer_output.jpg')
    vutils.save_image(output_grid, path, nrow=len(random_idx))

# Final Result
print(f"\nBest found score={best_score:.4f}")
logger.info(f"[BEST] iter {iter_num_best:03d} | Best found score={best_score:.4f} | Initial Best score={best_init_score:.4f}")
torch.save(best_point, os.path.join(output_directory, 'best_style_vector.pt')) 