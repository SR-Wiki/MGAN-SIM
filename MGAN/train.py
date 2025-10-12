from __future__ import print_function
import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch import tensor
# from torch.utils.data import DataLoader
from data_loader import DataLoader
import torch.backends.cudnn as cudnn
import torch.onnx
from networks import define_G, define_D, GANLoss, get_scheduler, update_learning_rate,SIMloss
from data import get_training_set, get_test_set
import numpy as np
import tifffile
import datetime

# Training settings
parser = argparse.ArgumentParser(description='MarkovianGAN-pytorch-implementation')
parser.add_argument('--sample_interval', type=int, default=100, help='test batch')
parser.add_argument('--dataset', type=str, default= 'Example', help='training dataset')
parser.add_argument('--discriminator', type=str, default= 'n_layers16', help='Markovian discriminator for training')
parser.add_argument('--save_path', type=str, default='./result/', help='the path to store the test results')
parser.add_argument('--batch_size', type=int, default=32, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--direction', type=str, default='a2b', help='a2b or b2a')
parser.add_argument('--input_nc', type=int, default=1, help='input image channels')
parser.add_argument('--output_nc', type=int, default=1, help='output image channels')
parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='discriminator filters in first conv layer')
parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count')
parser.add_argument('--n_epochs', type=int, default=50, help='# of iter at starting learning rate')
parser.add_argument('--n_epochs_decay', type=int, default=50, help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy: lambda|step|plateau|cosine')
parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', default='true', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--lamb', type=int, default=100, help='weight on L1 term in objective')
opt = parser.parse_args()
print(opt)

start_time = datetime.datetime.now()
if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

cudnn.benchmark = True
torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')

root_path = "dataset/"
train_set = get_training_set(root_path + opt.dataset)
test_set = get_test_set(root_path + opt.dataset)

data_loader1 = DataLoader(dataset_name = train_set)
data_loader2 = DataLoader(dataset_name = test_set)

device = torch.device("cuda" if opt.cuda else "cpu")

print('===> Building models')
net_g = define_G(opt.input_nc, opt.output_nc, opt.ngf, 'unet_128', 'batch', True, 'normal', 0.02, gpu_ids=[0,1])
net_d = define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.discriminator, gpu_ids =[0])

# setup optimizer
optimizer_g = optim.Adam(net_g.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizer_d = optim.Adam(net_d.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
net_g_scheduler = get_scheduler(optimizer_g, opt)
net_d_scheduler = get_scheduler(optimizer_d, opt)

criterionGAN = GANLoss("lsgan").to(device)
criterionL1 = nn.L1Loss().to(device)
criterionMSE = nn.MSELoss().to(device)
SIMloss = SIMloss()

os.makedirs(opt.save_path, exist_ok=True)

for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):

    # train
    for batch_i, batch in enumerate(data_loader1.load_batch(train_batch_size=opt.batch_size), 1):
        # forward

        real_a, real_b = torch.from_numpy(batch[1]).to(device), torch.from_numpy(batch[0]).to(device)

        fake_b = net_g(real_a)

        ######################
        # (1) Update D network
        ######################

        optimizer_d.zero_grad()

        # train with fake
        fake_ab = torch.cat((real_a, fake_b), 1)

        pred_fake = net_d.forward(fake_ab.detach())
        loss_d_fake = criterionGAN(pred_fake, False)
        # train with real
        real_ab = torch.cat((real_a, real_b), 1)
        pred_real = net_d.forward(real_ab)
        loss_d_real = criterionGAN(pred_real, True)
        # Combined D loss
        loss_d = (loss_d_fake + loss_d_real) * 0.5

        loss_d.backward()
        optimizer_d.step()

        ######################
        # (2) Update G network
        ######################

        optimizer_g.zero_grad()

        # First, G(A) should fake the discriminator
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = net_d.forward(fake_ab)
        loss_g_gan = criterionGAN(pred_fake, True)

        # Second, G(A) = B
        loss_g_ffl = SIMloss(fake_b, real_b)
        loss_g = loss_g_gan + loss_g_ffl * opt.lamb
        loss_g.backward()

        optimizer_g.step()

        elapsed_time = datetime.datetime.now() - start_time
        print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f} time: {}" .format(
            epoch, batch_i, data_loader1.batch_num, loss_d.item(), loss_g.item(), elapsed_time))

        if epoch % 10 == 0 and batch_i % opt.sample_interval == 0:

            for i in range(1, 2):
                for interval, img in enumerate(data_loader2.load_data(start=i, end=i + 1, is_testing=True), 1):

                    imgs_B, imgs_A = torch.from_numpy(img[1]).to(device), torch.from_numpy(img[0]).to(device)

                    fake_A = net_g(imgs_B)
                    fake_A = fake_A.cpu().detach().numpy()
                    imgs_A = imgs_A.cpu().detach().numpy()
                    imgs_B = imgs_B.cpu().detach().numpy()
                    fake_A = fake_A - np.min(fake_A)
                    fake_A = fake_A / np.max(fake_A)
                    imgs_A = imgs_A / np.max(imgs_A)
                    imgs_B = imgs_B / np.max(imgs_B)

                    save1_ = np.vstack((imgs_B, fake_A, imgs_A))
                    save1_ = 255 * np.array(save1_)
                    tifffile.imwrite(opt.save_path + '{}_{}_{}.tif'.format(epoch,batch_i,i), save1_.astype('uint8'))


    update_learning_rate(net_g_scheduler, optimizer_g)
    update_learning_rate(net_d_scheduler, optimizer_d)

    # test
    # avg_psnr = 0
    # for batch in enumerate(data_loader.load_data(batch_size=opt.batch_size)):
    #     input, target = batch[0].to(device), batch[1].to(device)
    #
    #     predictio = net_g(input)
    #     mse = criterionMSE(prediction, target)
    #     psnr = 10 * nlog10(1 / mse.item())
    #     avg_psnr += psnr
    # print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))

    #checkpoint
    if epoch % 10 == 0:
        if not os.path.exists("checkpoint"):
            os.mkdir("checkpoint")
        if not os.path.exists(os.path.join("checkpoint", opt.dataset)):
            os.mkdir(os.path.join("checkpoint", opt.dataset))

        net_g_model_out_path = "checkpoint/{}/netG_model_epoch_{}.pth".format(opt.dataset, epoch)
        net_d_model_out_path = "checkpoint/{}/netD_model_epoch_{}.pth".format(opt.dataset, epoch)
        torch.save(net_g, net_g_model_out_path)
        torch.save(net_d, net_d_model_out_path)
        print("Checkpoint saved to {}".format("checkpoint" + opt.dataset))
