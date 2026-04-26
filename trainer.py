"""
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from networks import AdaINGen, MsImageDis, VAEGen
from utils import weights_init, get_model_list, get_scheduler
from torch.autograd import Variable
import torch
import torch.nn as nn
import os
import pdb

class Generator_Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(Generator_Trainer, self).__init__()
        lr = hyperparameters['lr']

        # Initiate the networks
        self.gen = AdaINGen(hyperparameters['input_dim_a'], hyperparameters['gen'])  # auto-encoder for domain a
        self.dis = MsImageDis(hyperparameters['input_dim_a'], hyperparameters['dis'])  # discriminator for domain a
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        self.style_dim = hyperparameters['gen']['style_dim']

        # fix the noise used in sampling
        display_size = int(hyperparameters['display_size'])
        self.s_a = torch.randn(display_size, self.style_dim, 1, 1).cuda()
        self.s_b = torch.randn(display_size, self.style_dim, 1, 1).cuda()

        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        dis_params = self.dis.parameters()
        gen_params = self.gen.parameters()
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)

        # Network weight initialization
        self.apply(weights_init(hyperparameters['init']))
        self.dis.apply(weights_init('gaussian'))

        # # Load VGG model if needed
        # if 'vgg_w' in hyperparameters.keys() and hyperparameters['vgg_w'] > 0:
        #     self.vgg = load_vgg16(hyperparameters['vgg_model_path'] + '/models')
        #     self.vgg.eval()
        #     for param in self.vgg.parameters():
        #         param.requires_grad = False

        self.recon_criterion_cosine = torch.nn.CosineEmbeddingLoss()

    def recon_criterion_l1(self, input, target):
        return torch.mean(torch.abs(input - target))

    def forward(self, x_a, x_b):
        self.eval()
        s_a = Variable(self.s_a)
        s_b = Variable(self.s_b)
        c_a, s_a_fake = self.gen.encode(x_a)
        c_b, s_b_fake = self.gen.encode(x_b)
        x_ba = self.gen.decode(c_b, s_a)
        x_ab = self.gen.decode(c_a, s_b)
        self.train()
        return x_ab, x_ba


    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)
    

    def compute_kl_loss(self, style_vector):
        # Compute mean and log variance from style_vector
        mu = style_vector.mean(dim=(2, 3))  # (batch, style_dim)
        log_var = torch.log(style_vector.var(dim=(2, 3), unbiased=False) + 1e-6)

        # Compute KL divergence loss
        kl = 0.5 * torch.sum(mu**2 + torch.exp(log_var) - log_var - 1, dim=1)  # (batch,)
        return kl.mean()  # Batch-wise mean
    

    def sample_nomask(self, x_a, x_b, msk):
        self.eval()
        s_a1 = Variable(self.s_a)
        s_b1 = Variable(self.s_b)

        x_a_recon, x_b_recon, x_ba1, x_ba2, x_ab1, x_ab2 = [], [], [], [], [], []
        for i in range(x_a.size(0)):
            c_a, s_a_fake = self.gen.encode(x_a[i].unsqueeze(0))
            c_b, s_b_fake = self.gen.encode(x_b[i].unsqueeze(0))
            x_a_recon.append(self.gen.decode(c_a, s_a_fake) * msk[i].unsqueeze(0))
            x_b_recon.append(self.gen.decode(c_b, s_b_fake) * msk[i].unsqueeze(0))
            x_ba1.append(self.gen.decode(c_b, s_a1[i].unsqueeze(0)) * msk[i].unsqueeze(0))
            x_ba2.append(self.gen.decode(c_b, s_a_fake.unsqueeze(0)) * msk[i].unsqueeze(0))

            x_ab1.append(self.gen.decode(c_a, s_b1[i].unsqueeze(0)) * msk[i].unsqueeze(0))
            x_ab2.append(self.gen.decode(c_a, s_b_fake.unsqueeze(0)) * msk[i].unsqueeze(0))
        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ba1, x_ba2 = torch.cat(x_ba1), torch.cat(x_ba2)
        x_ab1, x_ab2 = torch.cat(x_ab1), torch.cat(x_ab2)
        self.train()
        return x_a * msk, x_a_recon, x_ab1, x_ab2, x_b * msk, x_b_recon, x_ba1, x_ba2


    def dis_update(self, x_a, x_b, msk, msk_toggle, hyperparameters):
        self.dis_opt.zero_grad()
        s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        # encode
        c_a, _ = self.gen.encode(x_a)
        c_b, _ = self.gen.encode(x_b)
        # decode (cross domain)
        x_ba = self.gen.decode(c_b, s_a) * msk if msk_toggle is True else self.gen.decode(c_b, s_a)
        x_ab = self.gen.decode(c_a, s_b) * msk if msk_toggle is True else self.gen.decode(c_a, s_b)
        # D loss
        self.loss_dis_a = self.dis.calc_dis_loss_new(x_ba.detach(), x_a)
        self.loss_dis_b = self.dis.calc_dis_loss_new(x_ab.detach(), x_b)
        self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis_a + hyperparameters['gan_w'] * self.loss_dis_b

        self.loss_dis_total.backward()
        self.dis_opt.step()
        return self.loss_dis_total
    

    def gen_update(self, x_a, x_b, msk, msk_toggle, hyperparameters):
        self.gen_opt.zero_grad()
        s_sample = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        # encode
        c_a, s_a = self.gen.encode(x_a)
        c_b, s_b = self.gen.encode(x_b)

        # decode (within domain)
        x_a_recon = self.gen.decode(c_a, s_a) * msk if msk_toggle is True else self.gen.decode(c_a, s_a)
        x_b_recon = self.gen.decode(c_b, s_b) * msk if msk_toggle is True else self.gen.decode(c_b, s_b)

        # decode (cross domain)
        x_ab = self.gen.decode(c_a, s_b) * msk if msk_toggle is True else self.gen.decode(c_a, s_b)
        x_ba = self.gen.decode(c_b, s_a) * msk if msk_toggle is True else self.gen.decode(c_b, s_a)

        # decode (new domain)
        x_a_sample = self.gen.decode(c_a, s_sample) * msk if msk_toggle is True else self.gen.decode(c_a, s_sample)
        x_b_sample = self.gen.decode(c_b, s_sample) * msk if msk_toggle is True else self.gen.decode(c_b, s_sample)

        # encode again
        c_a_recon, s_b_recon = self.gen.encode(x_ab)
        c_b_recon, s_a_recon = self.gen.encode(x_ba)

        # decode again
        x_aba = self.gen.decode(c_a_recon, s_a) * msk if msk_toggle is True else self.gen.decode(c_a_recon, s_a)
        x_bab = self.gen.decode(c_b_recon, s_b) * msk if msk_toggle is True else self.gen.decode(c_b_recon, s_b)

        # disentangle loss
        loss_gen_content = self.recon_criterion_l1(c_a, c_b)

        # reconstruction loss
        loss_gen_recon_x_a = self.recon_criterion_l1(x_a_recon, x_a)
        loss_gen_recon_x_b = self.recon_criterion_l1(x_b_recon, x_b)

        loss_gen_recon_x_ab = self.recon_criterion_l1(x_ab, x_b)
        loss_gen_recon_x_ba = self.recon_criterion_l1(x_ba, x_a)
        
        loss_gen_recon_c_a = self.recon_criterion_l1(c_a_recon, c_a)
        loss_gen_recon_c_b = self.recon_criterion_l1(c_b_recon, c_b)

        loss_gen_recon_s_a = self.recon_criterion_l1(s_a_recon, s_a)
        loss_gen_recon_s_b = self.recon_criterion_l1(s_b_recon, s_b)

        loss_gen_cycrecon_x_a = self.recon_criterion_l1(x_aba, x_a) if hyperparameters['recon_x_cyc_w'] > 0 else 0
        loss_gen_cycrecon_x_b = self.recon_criterion_l1(x_bab, x_b) if hyperparameters['recon_x_cyc_w'] > 0 else 0

        loss_gen_sample_x = self.recon_criterion_l1(x_a_sample, x_b_sample)

        # added GAN loss
        loss_gen_adv_ab = self.dis.calc_gen_loss_new(x_ab)
        loss_gen_adv_ba = self.dis.calc_gen_loss_new(x_ba)
        loss_gen_adv_a = self.dis.calc_gen_loss_new(x_a_recon)
        loss_gen_adv_b = self.dis.calc_gen_loss_new(x_b_recon)
        loss_gen_adv_sample_a = self.dis.calc_gen_loss_new(x_a_sample)
        loss_gen_adv_sample_b = self.dis.calc_gen_loss_new(x_b_sample)

        # added KL divergence loss
        loss_gen_kl_a = self.compute_kl_loss(s_a)
        loss_gen_kl_b = self.compute_kl_loss(s_b)
        loss_gen_kl_a_recon = self.compute_kl_loss(s_a_recon)
        loss_gen_kl_b_recon = self.compute_kl_loss(s_b_recon)

        # added latent regression loss
        c_a_sample_recon, s_a_sample_recon = self.gen.encode(x_a_sample)
        c_b_sample_recon, s_b_sample_recon = self.gen.encode(x_b_sample)
        loss_latent_reg_a = self.recon_criterion_l1(s_a_sample_recon, s_sample)
        loss_latent_reg_b = self.recon_criterion_l1(s_b_sample_recon, s_sample)
        loss_gen_recon_c_a_sample = self.recon_criterion_l1(c_a_sample_recon, c_a)
        loss_gen_recon_c_b_sample = self.recon_criterion_l1(c_b_sample_recon, c_b)

        x_a_recon_sample = self.gen.decode(c_a_sample_recon, s_a)
        x_b_recon_sample = self.gen.decode(c_b_sample_recon, s_b)
        loss_gen_recon_x_a_sample = self.recon_criterion_l1(x_a_recon_sample, x_a)
        loss_gen_recon_x_b_sample = self.recon_criterion_l1(x_b_recon_sample, x_b)


        # total loss
        self.loss_gen_total = hyperparameters['recon_c_w'] * loss_gen_content + \
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

        self.loss_gen_total.backward()
        self.gen_opt.step()
        return self.loss_gen_total


    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()


    def resume(self, checkpoint_dir, model_name, hyperparameters):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen_best_" + model_name)
        state_dict = torch.load(last_model_name)
        self.gen.load_state_dict(state_dict['gen'])
        iterations = int(last_model_name[-6:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis_best_" + model_name)
        state_dict = torch.load(last_model_name)
        self.dis.load_state_dict(state_dict['dis'])
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer_best_' + model_name))
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])
        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations


    def reload(self, checkpoint_dir, model_name):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen_best_" + model_name)
        state_dict = torch.load(last_model_name)
        self.gen.load_state_dict(state_dict['gen'])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis_best_" + model_name)
        state_dict = torch.load(last_model_name)
        self.dis.load_state_dict(state_dict['dis'])

        self.gen_scheduler = None
        self.dis_scheduler = None

    def reload_genonly(self, checkpoint_dir, model_name):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen_best_" + model_name)
        state_dict = torch.load(last_model_name)
        self.gen.load_state_dict(state_dict['gen'])
        # Load discriminators
        # last_model_name = get_model_list(checkpoint_dir, "dis_best_" + model_name)
        # state_dict = torch.load(last_model_name)
        # self.dis.load_state_dict(state_dict['dis'])

        self.gen_scheduler = None
        # self.dis_scheduler = None

    def save(self, snapshot_dir, iterations, tag, best=False):
        # Save generators, discriminators, and optimizers
        if best is True:
            gen_name = os.path.join(snapshot_dir, 'gen_best_' + tag + '_%03d.pt' % (iterations + 1))
            dis_name = os.path.join(snapshot_dir, 'dis_best_' + tag + '_%03d.pt' % (iterations + 1))
            opt_name = os.path.join(snapshot_dir, 'optimizer_best_' + tag + '_%03d.pt' % (iterations + 1))
            torch.save({'gen': self.gen.state_dict()}, gen_name)
            torch.save({'dis': self.dis.state_dict()}, dis_name)
            torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)
        else:
            gen_name = os.path.join(snapshot_dir, 'gen_%03d.pt' % (iterations + 1))
            dis_name = os.path.join(snapshot_dir, 'dis_%03d.pt' % (iterations + 1))
            opt_name = os.path.join(snapshot_dir, 'optimizer_%03d.pt' % (iterations + 1))
            torch.save({'gen': self.gen.state_dict()}, gen_name)
            torch.save({'dis': self.dis.state_dict()}, dis_name)
            torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)
