"""
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from utils import get_data_loader, prepare_sub_folder, write_html, get_config, write_2images, Timer, calculate_metric, setting_logger, genloss_sample
import argparse
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
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torchvision.utils as vutils

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_num', type=str, default='0')
parser.add_argument('--config', type=str, default='configs/bboxharmony_generator.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='checkpoints/temp_generator/', help="outputs path")
parser.add_argument("--resume", action="store_true")
parser.add_argument('--normalize_type', type=str, default='percentile', help="normalization type (percentile|minmax)")
parser.add_argument('--mask_toggle', type=bool, default=True, help="True|False")
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
os.environ["CUDA_VISIBLE_DEVICES"]= opts.gpu_num
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
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder

'''
Setup dataloader
'''
train_loader_a = get_data_loader(logger, config, disentangle=True, zo_opt=False, phase='train', normalize=opts.normalize_type)
test_loader_a = get_data_loader(logger, config, disentangle=True, zo_opt=False, phase='val', normalize=opts.normalize_type)

'''
Setup disentangle model
'''
trainer = Generator_Trainer(config)
trainer.to(device)

'''
Setup display images
'''
train_display_images_a = torch.stack([train_loader_a.dataset[i][0] for i in range(display_size)]).to(device)
train_display_images_b = torch.stack([train_loader_a.dataset[i][1] for i in range(display_size)]).to(device)
train_display_mask = torch.stack([train_loader_a.dataset[i][2] for i in range(display_size)]).to(device)
test_display_images_a = torch.stack([test_loader_a.dataset[i][0] for i in range(display_size)]).to(device)
test_display_images_b = torch.stack([test_loader_a.dataset[i][1] for i in range(display_size)]).to(device)
test_display_mask = torch.stack([test_loader_a.dataset[i][2] for i in range(display_size)]).to(device)


best_loss = np.inf
best_psnr = 0
best_ssim = 0

'''
training
'''
iterations = trainer.resume(checkpoint_directory, hyperparameters=config) if opts.resume else 0
while True:
    dis_loss = 0
    gen_loss = 0

    for images_a, images_b, mask in tqdm(train_loader_a):
        trainer.train()

        images_a, images_b, mask = images_a.to(device), images_b.to(device), mask.to(device)

        if opts.mask_toggle is True:
            images_a = images_a * mask
            images_b = images_b * mask

        images_a, images_b = images_a.to(device), images_b.to(device)
        dis_loss += trainer.dis_update(images_a, images_b, mask, opts.mask_toggle, config).item()
        gen_loss += trainer.gen_update(images_a, images_b, mask, opts.mask_toggle, config).item()

        trainer.update_learning_rate()

    logger.info("[%03d/%03d] Train) Gen loss: %.4f / Dis loss: %.4f" % (iterations+1, max_iter, gen_loss / len(train_loader_a.dataset), dis_loss / len(train_loader_a.dataset)))
    writer.add_scalar("Train loss/epoch", gen_loss / len(train_loader_a.dataset), iterations+1)

    # Validation
    with torch.no_grad():
        total_loss = 0
        total_psnr = 0
        total_ssim = 0

        for images_a, images_b, mask in test_loader_a:
            trainer.eval()
            images_a, images_b, mask = images_a.to(device).detach(), images_b.to(device).detach(), mask.to(device).detach()

            if opts.mask_toggle is True:
                images_a = images_a * mask
                images_b = images_b * mask

            gen_loss, images_a2b, images_b2a = genloss_sample(trainer.gen, trainer.dis, images_a, images_b, mask, opts.mask_toggle, config)
            total_loss += gen_loss.item()

            cur_psnr, cur_ssim = calculate_metric(images_a2b, images_b, mask)
            total_psnr += cur_psnr
            total_ssim += cur_ssim

            cur_psnr, cur_ssim = calculate_metric(images_b2a, images_a, mask)
            total_psnr += cur_psnr
            total_ssim += cur_ssim

    total_psnr = total_psnr / (len(test_loader_a.dataset) * 2)
    total_ssim = total_ssim / (len(test_loader_a.dataset) * 2)
    total_loss = total_loss / len(test_loader_a.dataset)

    logger.info('[%03d/%03d] Val) loss %.4f / psnr %.2f / ssim %.4f' % (iterations+1, max_iter, total_loss, total_psnr, total_ssim))
    writer.add_scalar("Valid loss/epoch", total_loss, iterations+1)
    writer.add_scalar("Valid psnr/epoch", total_psnr, iterations+1)
    writer.add_scalar("Valid ssim/epoch", total_ssim, iterations+1)

    # Best model saving 
    if total_loss < best_loss:
        logger.info('** loss best model saved !! **')
        best_loss = total_loss
        trainer.save(checkpoint_directory, tag='loss', iterations=iterations, best=True)
    if total_psnr > best_psnr:
        logger.info('** psnr best model saved !! **')
        best_psnr = total_psnr
        trainer.save(checkpoint_directory, tag='psnr', iterations=iterations, best=True)
    if total_ssim > best_ssim:
        logger.info('** ssim best model saved !! **')
        best_ssim = total_ssim
        trainer.save(checkpoint_directory, tag='ssim', iterations=iterations, best=True)

    iterations += 1

    # Write images
    with torch.no_grad():
        test_image_outputs = trainer.sample_nomask(test_display_images_a, test_display_images_b, test_display_mask)
        train_image_outputs = trainer.sample_nomask(train_display_images_a, train_display_images_b, train_display_mask)
    write_2images(test_image_outputs, display_size, image_directory, 'test_%03d' % (iterations + 1))
    write_2images(train_image_outputs, display_size, image_directory, 'train_%03d' % (iterations + 1))
    # HTML
    write_html(output_directory + "/index.html", iterations + 1, config['image_save_iter'], 'images')

    if iterations >= max_iter:
        sys.exit('Finish training')