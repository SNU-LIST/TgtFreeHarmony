"""
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from torch.utils.data import DataLoader
from networks import Vgg16
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import transforms
from data import ImagePickle_one_encdec, ImagePickle_pseudo_trg
import torch
import torch.nn as nn
import os
import math
import torchvision.utils as vutils
import yaml
import numpy as np
import torch.nn.init as init
import time
import datetime
import cv2
import logging
import logging_helper
import torch.nn.functional as F

# Methods
# get_all_data_loaders      : primary data loader interface (load trainA, testA, trainB, testB)
# get_data_loader_list      : list-based data loader
# get_data_loader_folder    : folder-based data loader
# get_config                : load yaml file
# eformat                   :
# write_2images             : save output image
# prepare_sub_folder        : create checkpoints and images folders for saving outputs
# write_one_row_html        : write one row of the html file for output images
# write_html                : create the html file.
# write_loss
# slerp
# get_slerp_interp
# get_model_list
# load_vgg16
# load_inception
# vgg_preprocess
# get_scheduler
# weights_init


def cmap_convert(N=256, normalized=False):
    colors = {
        0: [0, 0, 0],       # Background: Black
        1: [255, 0, 0],     # Class 1: Red
        2: [0, 255, 0],     # Class 2: Green
        3: [0, 0, 255]      # Class 3: Blue
    }
    return colors


def compute_kl_loss(style_vector):
    # Compute mean and log variance from style_vector
    mu = style_vector.mean(dim=(2, 3))  # (batch, style_dim)
    log_var = torch.log(style_vector.var(dim=(2, 3), unbiased=False) + 1e-6)

    # Compute KL divergence loss
    kl = 0.5 * torch.sum(mu**2 + torch.exp(log_var) - log_var - 1, dim=1)  # (batch,)
    return kl.mean()  # Batch-wise mean


def genloss_sample(gen, dis, x_a, x_b, msk, msk_toggle, hyperparameters):
    with torch.no_grad():
        gen.eval()
        dis.eval()

        style_dim = hyperparameters['gen']['style_dim']

        s_sample = Variable(torch.randn(x_a.size(0), style_dim, 1, 1).cuda())
        # encode
        c_a, s_a = gen.encode(x_a)
        c_b, s_b = gen.encode(x_b)

        # decode (within domain)
        x_a_recon = gen.decode(c_a, s_a) * msk if msk_toggle is True else gen.decode(c_a, s_a)
        x_b_recon = gen.decode(c_b, s_b) * msk if msk_toggle is True else gen.decode(c_b, s_b)

        # decode (cross domain)
        x_ab = gen.decode(c_a, s_b) * msk if msk_toggle is True else gen.decode(c_a, s_b)
        x_ba = gen.decode(c_b, s_a) * msk if msk_toggle is True else gen.decode(c_b, s_a)

        #decode (new sample)
        x_a_sample = gen.decode(c_a, s_sample) * msk if msk_toggle is True else gen.decode(c_a, s_sample)
        x_b_sample = gen.decode(c_b, s_sample) * msk if msk_toggle is True else gen.decode(c_b, s_sample)

        # encode again
        c_a_recon, s_b_recon = gen.encode(x_ab)
        c_b_recon, s_a_recon = gen.encode(x_ba)

        # decode again (if needed)
        x_aba = gen.decode(c_a_recon, s_a) * msk if msk_toggle is True else gen.decode(c_a_recon, s_a)
        x_bab = gen.decode(c_b_recon, s_b) * msk if msk_toggle is True else gen.decode(c_b_recon, s_b)

        # disentangle loss
        loss_gen_content = l1loss(c_a, c_b)

        # reconstruction loss
        loss_gen_recon_x_a = l1loss(x_a_recon, x_a)
        loss_gen_recon_x_b = l1loss(x_b_recon, x_b)

        loss_gen_recon_x_ab = l1loss(x_ab, x_b)
        loss_gen_recon_x_ba = l1loss(x_ba, x_a)

        loss_gen_recon_c_a = l1loss(c_a_recon, c_a)
        loss_gen_recon_c_b = l1loss(c_b_recon, c_b)

        loss_gen_recon_s_a = l1loss(s_a_recon, s_a)
        loss_gen_recon_s_b = l1loss(s_b_recon, s_b)

        loss_gen_cycrecon_x_a = l1loss(x_aba, x_a)
        loss_gen_cycrecon_x_b = l1loss(x_bab, x_b)

        loss_gen_sample_x = l1loss(x_a_sample, x_b_sample)

        # added GAN loss
        loss_gen_adv_ab = dis.calc_gen_loss_new(x_ab)
        loss_gen_adv_ba = dis.calc_gen_loss_new(x_ba)
        loss_gen_adv_a = dis.calc_gen_loss_new(x_a_recon)
        loss_gen_adv_b = dis.calc_gen_loss_new(x_b_recon)
        loss_gen_adv_sample_a = dis.calc_gen_loss_new(x_a_sample)
        loss_gen_adv_sample_b = dis.calc_gen_loss_new(x_b_sample)

        # added KL divergence loss
        loss_gen_kl_a = compute_kl_loss(s_a)
        loss_gen_kl_b = compute_kl_loss(s_b)
        loss_gen_kl_a_recon = compute_kl_loss(s_a_recon)
        loss_gen_kl_b_recon = compute_kl_loss(s_b_recon)

        # added latent regression loss
        c_a_sample_recon, s_a_sample_recon = gen.encode(x_a_sample)
        c_b_sample_recon, s_b_sample_recon = gen.encode(x_b_sample)
        loss_latent_reg_a = l1loss(s_a_sample_recon, s_sample)
        loss_latent_reg_b = l1loss(s_b_sample_recon, s_sample)
        loss_gen_recon_c_a_sample = l1loss(c_a_sample_recon, c_a)
        loss_gen_recon_c_b_sample = l1loss(c_b_sample_recon, c_b)

        x_a_recon_sample = gen.decode(c_a_sample_recon, s_a)
        x_b_recon_sample = gen.decode(c_b_sample_recon, s_b)
        loss_gen_recon_x_a_sample = l1loss(x_a_recon_sample, x_a)
        loss_gen_recon_x_b_sample = l1loss(x_b_recon_sample, x_b)
        
        # total loss
        loss_gen_total = hyperparameters['recon_c_w'] * loss_gen_content + \
                        hyperparameters['recon_x_w'] * loss_gen_recon_x_a + \
                        hyperparameters['recon_x_w'] * loss_gen_recon_x_b + \
                        hyperparameters['recon_x_w'] * loss_gen_recon_x_ab + \
                        hyperparameters['recon_x_w'] * loss_gen_recon_x_ba + \
                        hyperparameters['recon_c_w'] * loss_gen_recon_c_a + \
                        hyperparameters['recon_c_w'] * loss_gen_recon_c_b + \
                        hyperparameters['recon_s_w'] * loss_gen_recon_s_a + \
                        hyperparameters['recon_s_w'] * loss_gen_recon_s_b + \
                        hyperparameters['recon_x_cyc_w'] * loss_gen_cycrecon_x_a + \
                        hyperparameters['recon_x_cyc_w'] * loss_gen_cycrecon_x_b + \
                        hyperparameters['recon_x_w'] * loss_gen_sample_x + \
                        hyperparameters['gan_w'] * loss_gen_adv_ab + \
                        hyperparameters['gan_w'] * loss_gen_adv_ba + \
                        hyperparameters['gan_w'] * loss_gen_adv_a + \
                        hyperparameters['gan_w'] * loss_gen_adv_b + \
                        hyperparameters['gan_w'] * loss_gen_adv_sample_a + \
                        hyperparameters['gan_w'] * loss_gen_adv_sample_b + \
                        hyperparameters['kl_w'] * loss_gen_kl_a + \
                        hyperparameters['kl_w'] * loss_gen_kl_b + \
                        hyperparameters['kl_w'] * loss_gen_kl_a_recon + \
                        hyperparameters['kl_w'] * loss_gen_kl_b_recon + \
                        hyperparameters['recon_s_w'] * loss_latent_reg_a + \
                        hyperparameters['recon_s_w'] * loss_latent_reg_b + \
                        hyperparameters['recon_c_w'] * loss_gen_recon_c_a_sample + \
                        hyperparameters['recon_c_w'] * loss_gen_recon_c_b_sample + \
                        hyperparameters['recon_x_w'] * loss_gen_recon_x_a_sample + \
                        hyperparameters['recon_x_w'] * loss_gen_recon_x_b_sample
        
        return loss_gen_total, x_ab, x_ba


def l1loss(input, target):
    return torch.mean(torch.abs(input - target))


def setting_logger(path, file_name, module_type):
    if module_type == 'train':
        logger = logging.getLogger("module.train")
    elif module_type == 'test':
        logger = logging.getLogger("module.test")
        
    logger.setLevel(logging.INFO)
    logging_helper.setup(path, file_name)
    
    nowDate = datetime.datetime.now().strftime('%Y-%m-%d')
    nowTime = datetime.datetime.now().strftime('%H:%M:%S')
    logger.info('Date: ' + str(nowDate) + '  ' +  str(nowTime))

    return logger


def f1_score(mask1, mask2):
    # Flatten the tensors to compare pixel-wise
    mask1 = mask1.view(-1).float()
    mask2 = mask2.view(-1).float()

    # True positives, False positives, and False negatives
    tp = (mask1 * mask2).sum()  # True positives
    fp = ((1 - mask1) * mask2).sum()  # False positives
    fn = (mask1 * (1 - mask2)).sum()  # False negatives

    # Precision and Recall
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    # F1 Score
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    return f1.item()

       
def f1_score_multi(pred, target, num_classes, epsilon=1e-6):
    pred = pred.to(torch.int64)
    target = target.to(torch.int64)

    target = F.one_hot(target, num_classes).permute(0, 3, 1, 2)

    TP = torch.sum(pred * target, dim=(0, 2, 3))  # True Positive
    FP = torch.sum(pred * (1 - target), dim=(0, 2, 3))  # False Positive
    FN = torch.sum((1 - pred) * target, dim=(0, 2, 3))  # False Negative

    # Precision, Recall
    precision = (TP + epsilon) / (TP + FP + epsilon)
    recall = (TP + epsilon) / (TP + FN + epsilon)

    # F1 Score
    f1_scores = (2 * precision * recall) / (precision + recall + epsilon)

    return f1_scores


class SSIM_cal_2D(nn.Module):
    def __init__(self, win_size = 7, k1 = 0.01, k2 = 0.03):
        super(SSIM_cal_2D, self).__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size) / win_size ** 2)
        NP = win_size ** 2
        self.cov_norm = NP / (NP - 1)

    def forward(self, X, Y, data_range):
        data_range = data_range[:, None, None, None]
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        ux = F.conv2d(X, self.w.to(X.device))
        uy = F.conv2d(Y, self.w.to(X.device))
        uxx = F.conv2d(X * X, self.w.to(X.device))
        uyy = F.conv2d(Y * Y, self.w.to(X.device))
        uxy = F.conv2d(X * Y, self.w.to(X.device))
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux ** 2 + uy ** 2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / D

        return torch.mean(S,[2,3],keepdim=False)
ssim_cal_2d = SSIM_cal_2D()


def ssim(img1, img2, mask):
    img1 = np.squeeze(np.array(img1.cpu()))
    img2 = np.squeeze(np.array(img2.cpu()))
    mask = np.squeeze(np.array(mask.cpu()))

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mask = mask[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))

    return ssim_map[mask>0].mean()


def calculate_ssim(img, ref, mask):
    mask = mask#.to(img.device)
    img = img * mask
    ref = ref * mask

    img = img.unsqueeze(0).unsqueeze(0)
    ref = ref.unsqueeze(0).unsqueeze(0)
    
    ones = torch.ones(ref.shape[0]).to(ref.device)
    
    if len(img.shape) == 5:
        # ssim = ssim_cal_3d(img, ref, ones)
        print('error')
    elif len(img.shape) == 4:
        ssim = ssim_cal_2d(img, ref, ones)
    return ssim


def calculate_metric(output, label, mask):
    # Calculating psnr, ssim
    output = output * mask
    label = label * mask
    
    # mse = torch.mean((im1-im2)**2)
    mse = torch.mean((output[mask>0]-label[mask>0])**2)
    
    if mse == 0:
        psnr_value = 100
    else:
        #PIXEL_MAX = max(label[mask])
        PIXEL_MAX = 1
        psnr_value = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

    ssim_value = calculate_ssim(output.squeeze(), label.squeeze(), mask.squeeze())

    return psnr_value, ssim_value


def get_data_loader(logger, conf, disentangle, zo_opt, phase, normalize):
    data_loader = get_data_loader_pickle_oneencdec(logger, conf, disentangle, zo_opt, phase=phase, normalize=normalize)
        
    return data_loader


def get_data_loader_pickle_oneencdec(logger, conf, disentangle, zo_opt, phase, normalize):
    batch_size = conf['batch_size']
    num_workers = conf['num_workers']

    transform_list = [transforms.ToTensor()]
    transform = transforms.Compose(transform_list)

    if disentangle is True:
        dataset = ImagePickle_one_encdec(logger, conf, phase=phase, normalize=normalize, transform=transform)
    else:
        dataset = ImagePickle_pseudo_trg(logger, conf, zo_opt=zo_opt, phase=phase, normalize=normalize, transform=transform)

    if phase == 'train':
        loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    else:
        loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, drop_last=True, num_workers=num_workers)
    return loader


def get_config(path):
    with open(path, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)


def __write_images(image_outputs, display_image_num, file_name):
    image_outputs = [images.expand(-1, 3, -1, -1) for images in image_outputs] # expand gray-scale images to 3 channels
    image_tensor = torch.cat([images[:display_image_num] for images in image_outputs], 0)
    image_grid = vutils.make_grid(image_tensor.data, nrow=display_image_num, padding=0)#, normalize=True)
    vutils.save_image(image_grid, file_name, nrow=1)


def write_2images(image_outputs, display_image_num, image_directory, postfix):
    n = len(image_outputs)
    __write_images(image_outputs[0:n//2], display_image_num, '%s/gen_a2b_%s.jpg' % (image_directory, postfix))
    __write_images(image_outputs[n//2:n], display_image_num, '%s/gen_b2a_%s.jpg' % (image_directory, postfix))


def prepare_sub_folder(output_directory):
    image_directory = os.path.join(output_directory, 'images')
    if not os.path.exists(image_directory):
        print("Creating directory: {}".format(image_directory))
        os.makedirs(image_directory)
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)
    return checkpoint_directory, image_directory


def write_one_row_html(html_file, iterations, img_filename, all_size):
    html_file.write("<h3>iteration [%d] (%s)</h3>" % (iterations,img_filename.split('/')[-1]))
    html_file.write("""
        <p><a href="%s">
          <img src="%s" style="width:%dpx">
        </a><br>
        <p>
        """ % (img_filename, img_filename, all_size))
    return


def write_html(filename, iterations, image_save_iterations, image_directory, all_size=1536):
    html_file = open(filename, "w")
    html_file.write('''
    <!DOCTYPE html>
    <html>
    <head>
      <title>Experiment name = %s</title>
      <meta http-equiv="refresh" content="30">
    </head>
    <body>
    ''' % os.path.basename(filename))
    html_file.write("<h3>current</h3>")
    write_one_row_html(html_file, iterations, '%s/gen_a2b_train_current.jpg' % (image_directory), all_size)
    write_one_row_html(html_file, iterations, '%s/gen_b2a_train_current.jpg' % (image_directory), all_size)
    for j in range(iterations, image_save_iterations-1, -1):
        if j % image_save_iterations == 0:
            write_one_row_html(html_file, j, '%s/gen_a2b_test_%08d.jpg' % (image_directory, j), all_size)
            write_one_row_html(html_file, j, '%s/gen_b2a_test_%08d.jpg' % (image_directory, j), all_size)
            write_one_row_html(html_file, j, '%s/gen_a2b_train_%08d.jpg' % (image_directory, j), all_size)
            write_one_row_html(html_file, j, '%s/gen_b2a_train_%08d.jpg' % (image_directory, j), all_size)
    html_file.write("</body></html>")
    html_file.close()


def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pt" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name


# def load_vgg16(model_dir):
#     """ Use the model from https://github.com/abhiskk/fast-neural-style/blob/master/neural_style/utils.py """
#     if not os.path.exists(model_dir):
#         os.mkdir(model_dir)
#     if not os.path.exists(os.path.join(model_dir, 'vgg16.weight')):
#         if not os.path.exists(os.path.join(model_dir, 'vgg16.t7')):
#             os.system('wget https://www.dropbox.com/s/76l3rt4kyi3s8x7/vgg16.t7?dl=1 -O ' + os.path.join(model_dir, 'vgg16.t7'))
#         vgglua = load_lua(os.path.join(model_dir, 'vgg16.t7'))
#         vgg = Vgg16()
#         for (src, dst) in zip(vgglua.parameters()[0], vgg.parameters()):
#             dst.data[:] = src
#         torch.save(vgg.state_dict(), os.path.join(model_dir, 'vgg16.weight'))
#     vgg = Vgg16()
#     vgg.load_state_dict(torch.load(os.path.join(model_dir, 'vgg16.weight')))
#     return vgg


# def load_inception(model_path):
#     state_dict = torch.load(model_path)
#     model = inception_v3(pretrained=False, transform_input=True)
#     model.aux_logits = False
#     num_ftrs = model.fc.in_features
#     model.fc = nn.Linear(num_ftrs, state_dict['fc.weight'].size(0))
#     model.load_state_dict(state_dict)
#     for param in model.parameters():
#         param.requires_grad = False
#     return model


# def vgg_preprocess(batch):
#     tensortype = type(batch.data)
#     batch = torch.cat((batch, batch, batch), dim = 1) # convert gray image to 3 channels
#     batch = (batch) * 255 if np.max(batch) <= 1.0 else batch # [0, 1] -> [0, 255]
#     # Is noramlization needed?
#     # mean = tensortype(batch.data.size()).cuda()
#     # mean[:, 0, :, :] = 103.939
#     # mean[:, 1, :, :] = 116.779
#     # mean[:, 2, :, :] = 123.680
#     # batch = batch.sub(Variable(mean)) # subtract mean
#     return batch


def get_scheduler(optimizer, hyperparameters, iterations=-1):
    if 'lr_policy' not in hyperparameters or hyperparameters['lr_policy'] == 'constant':
        scheduler = None # constant scheduler
    elif hyperparameters['lr_policy'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=hyperparameters['step_size'],
                                        gamma=hyperparameters['gamma'], last_epoch=iterations)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', hyperparameters['lr_policy'])
    return scheduler


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun


class Timer:
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(self.msg % (time.time() - self.start_time))