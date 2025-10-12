from __future__ import print_function
import argparse
import os
import time
import tifffile
import torch
import scipy
import numpy as np
import percent

# inference settings
parser = argparse.ArgumentParser(description='MarkovianGAN-pytorch-implementation')
parser.add_argument('--dataset', required = True, type=str, help='Microtubules')
parser.add_argument('--data_path', required = True, type=str, default='./input.tif', help='input image for inference')
# parser.add_argument('--save_path', required = True, type=str, default='./result.tif', help='inference result')
parser.add_argument('--nepochs', type=int, default=100, help='saved model of which epochs')
parser.add_argument('--cuda', default='true', action='store_true', help='use cuda')
opt = parser.parse_args()
print(opt)

device = torch.device("cuda:0" if opt.cuda else "cpu")

model_path = "checkpoint/{}/netG_model_epoch_{}.pth".format(opt.dataset, opt.nepochs)
net_g = torch.load(model_path).to(device)

img_stack = tifffile.imread(opt.data_path).astype('float32')
if len(img_stack.shape) == 2:
    img_stack = img_stack[np.newaxis, :, :]  # 新shape: (1, h, w)
elif len(img_stack.shape) == 3:
    pass
else:
    raise ValueError(f"不支持的图像维度：{img_stack.shape}（仅支持2D/3D TIFF）")
[t,h,w] = img_stack.shape

imsize = (t,h,w)
result = np.zeros(imsize)
if not os.path.exists("prediction"):
    os.mkdir("prediction")
for i in range(0, t):
    # img = percent.normalize_percentage(img_stack[i], 0.3, 99.97)
    img = img_stack[i]
    img = img - np.min(img)
    img = img / np.max(img)
    imgs = []

    img = img.reshape(1, h, w)
    imgs.append(img)
    imgs_B = np.array(imgs)
    imgs_B = torch.from_numpy(imgs_B).to(device)

    fake_A = net_g(imgs_B)
    fake_A = fake_A.cpu().detach().numpy()
    fake_A = fake_A - np.min(fake_A)
    fake_A = fake_A / np.max(fake_A)
    save_ = 255 * np.array(fake_A)
    result[i] = save_
tifffile.imwrite("prediction/result.tif", result.astype('uint16'))


