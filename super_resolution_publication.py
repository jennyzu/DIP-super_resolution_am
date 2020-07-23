from __future__ import print_function
import matplotlib.pyplot as plt

import argparse
import os
import numpy as np
import cv2
import re
import self as self
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from PerceptualSimilarity import compute_dists
from scipy import ndimage

import torch
import torch.optim
import torch.nn.functional as F
from typing import Union

from models import *
from utils.sr_utils import *
from utils.sr_filter import *
#import RunMe_super_resolution
import load_images_publication

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def imwrite_multi_tiff(Images: Union[list, np.ndarray], output_filename, Norm_frames=True, resize_factors=[1]):
    # Writes a multi page (multi-frame) tiff file
    # Inputs:
    # Images can be either a list of ndarrays or single ndarray
    #
    # Output:
    #     tiff file will be either created ar images added to existing tiff file
    #
    # !!!! output_filename --> file name must be "tiff"
    #
    assert output_filename is not None, "Must Provide image file Name"
    filename, file_extension = os.path.splitext(output_filename)
    assert file_extension.lower() == '.tiff', 'error - file extention must be .tiff'

    if type(Images) is np.ndarray:
        Images = [Images]

    if not type(resize_factors) is list:
        resize_factors = [resize_factors]
    if len(resize_factors) < len(Images):
        resize_factors *= len(Images)

    imlist = []
    for resize_factor, Frame in zip(resize_factors, Images):
        assert type(Frame) is np.ndarray, "imwrite_multi_tiff expects only ndarray within list items "
        if Norm_frames:
            Frame = ((Frame - Frame.min()) / (Frame.max() - Frame.min()) * 255).astype(np.uint8)
            # Frame = (Frame * 65535).astype(np.uint16)
            #Frame = (Frame * 255).astype(np.uint8)

        if len(Frame.shape) == 3 and Frame.shape[2] == 3:
            imlist.append(Image.fromarray(Frame, mode="RGB"))
        else:
            imlist.append(Image.fromarray(Frame))

        if not resize_factor == 1:
            imlist[-1] = imlist[-1].resize([int(resize_factor * s) for s in imlist[-1].size])

    imlist[0].save(output_filename, save_all=True, append_images=imlist[1:])

    # print(Image.open("test.tiff").n_frames) # - Get number of frames

def custom_downsample(HR_img, scale, use_5_per):
    LR_img = downsample_custom(HR_img, scale, self.device, use_5_per)
    return LR_img

def pad_shift_filter(signal, filter):
    pad_x = signal.shape[2] - filter.shape[2]
    pad_y = signal.shape[3] - filter.shape[3]
    expanded_kernel = F.pad(filter, [0, pad_y, 0, pad_x])
    expanded_kernel_np = expanded_kernel.cpu().numpy()
    expanded_kernel_shift = np.roll(expanded_kernel_np, -int(filter.shape[2] / 2), axis=2)
    expanded_kernel_shift = np.roll(expanded_kernel_shift, -int(filter.shape[2] / 2), axis=3)
    expanded_kernel_shift = torch.from_numpy(expanded_kernel_shift).float()

    return expanded_kernel_shift

def torch_fourier_conv(f, k):
    ### fft of h*x
    expanded_kernel_shift = pad_shift_filter(f, k)
    fft_hx = torch.empty([f.shape[0], f.shape[1], f.shape[2], f.shape[3], 2])
    for i in range(3):
        fft_x = torch.rfft(f[:, i:i + 1, :, :], 2, onesided=False, normalized=True).to(self.device)
        fft_kernel = torch.rfft(expanded_kernel_shift, 2, onesided=False, normalized=True).to(self.device)
        real = fft_x[:, :, :, :, 0] * fft_kernel[:, :, :, :, 0] - \
               fft_x[:, :, :, :, 1] * fft_kernel[:, :, :, :, 1]
        im = fft_x[:, :, :, :, 0] * fft_kernel[:, :, :, :, 1] + \
             fft_x[:, :, :, :, 1] * fft_kernel[:, :, :, :, 0]
        fft_conv = torch.stack([real, im], -1)  # (a+bj)*(c+dj) = (ac-bd)+(ad+bc)j
        fft_hx[:, i, :, :, :] = fft_conv

    return fft_kernel, fft_hx

def mse_fft(input, target, size_average=True):
    L = (input - target)
    L_fft = torch.rfft(L, 2, onesided=False, normalized=True).to(self.device)
    L_fft = L_fft ** 2
    return torch.mean(L_fft) if size_average else torch.sum(L_fft)

def mse_bp(hx, y, scale):
    # first part: LS loss
    dip_loss_dif = (hx - y)
    dip_loss = torch.rfft(dip_loss_dif, 2, onesided=False, normalized=True).to(self.device)

    # second part: BP loss
    eps = 1e-3
    mul_factor = 1e5
    eps_ignored = 0.01
    sigma = 0
    h = torch.from_numpy(get_bicubic(scale))
    h = torch.unsqueeze(h, 0).unsqueeze(0)
    conv_shape = (h.shape[2] + h.shape[2] - 1, h.shape[3] + h.shape[3] - 1)
    H = fft2(h, conv_shape[1], conv_shape[0])
    H_flip = fft2(flip(h), conv_shape[1], conv_shape[0])
    H_mul_H_flip = mul_complex(H, H_flip)
    H_mul_H_flip_ifft = torch.irfft(H_mul_H_flip, signal_ndim=2, normalized=True, onesided=False)
    h_downsampled = H_mul_H_flip_ifft[:,:,1::scale, 1::scale]
    h_downsampled = pad_shift_filter(y, h_downsampled)

    H_downsampled = torch.rfft(h_downsampled, signal_ndim=2, normalized=True, onesided=False)
    bp_loss = torch.sqrt(abs2(H_downsampled)[:,:,:,:,0:1])
    bp_loss = mul_factor * bp_loss + eps_ignored * (sigma ** 2) + eps
    bp_loss = 1 / (torch.sqrt(bp_loss))
    bp_loss = torch.repeat_interleave(bp_loss, 2, -1).to(self.device)
    loss_mat = bp_loss.to(self.device) * dip_loss

    return torch.mean(loss_mat ** 2)

def dip_sr(img_name, loss_type, directory, pix_ignore, factor, imgs_dir, use_5_per):

    img_name_for_plot = img_name[0:-5]
    print(img_name_for_plot)
    learning_rate = 1e-3
    OPTIMIZER = 'adam'
    if factor == 3:
        num_iter = 2000
        reg_noise_std = 0.03
    if factor == 5:
        num_iter = 600
        reg_noise_std = 0.03

    tv_weight = 0
    PLOT = True
    PLOT_PSNR = True

    path_to_image = imgs_dir + img_name
    low_res, high_res = load_images_publication.tiff_imgs(path_to_image, factor, use_5_per)

    # sanity check
    #low_res = create_lr_image.create(high_res)
    if use_5_per ==False:
        high_res_pil = Image.fromarray(np.uint8(high_res*255), 'L')
        low_res_pil = Image.fromarray(np.uint8(low_res*255), 'L')
        img_bicubic, img_sharp, img_nearest = get_baselines(low_res_pil, high_res_pil)
        high_res = np.expand_dims(high_res, axis=0)
        low_res_for_up = torch.from_numpy(low_res)
        low_res_for_up = torch.unsqueeze(low_res_for_up, 0).unsqueeze(0)
        img_upsampled_torch, img_bicubic1 = upsample_using_h(low_res_for_up, factor, self.device)
        img_upsampled = img_upsampled_torch.cpu().numpy()
        img_upsampled = np.squeeze(img_upsampled, axis=0)
        img_upsampled = (img_upsampled - img_upsampled.min()) / (img_upsampled.max() - img_upsampled.min())
    else:
        img_bicubic = np.zeros((5, high_res.shape[1], high_res.shape[2]))
        img_nearest= np.zeros((5, high_res.shape[1], high_res.shape[2]))
        for j in range(5):
            img_bicubic[j,:,:] = cv2.resize(low_res[j,:,:], dsize=high_res.shape[1:], interpolation=cv2.INTER_CUBIC)
            img_nearest[j,:,:] = cv2.resize(low_res[j,:,:], dsize=high_res.shape[1:], interpolation=cv2.INTER_NEAREST)

    high_res_ = high_res

    psnr_bicubic = compare_psnr(high_res, img_bicubic, data_range=1)
    psnr_nn = compare_psnr(high_res, img_nearest, data_range=1)
    psnr_custom = compare_psnr(high_res, img_upsampled, data_range=1)
    if use_5_per == False:
        perc_sim_bicubic = compute_dists.compute(high_res[0,:,:], img_bicubic[0,:,:])
        perc_sim_bicubic = perc_sim_bicubic.cpu().detach().numpy()[0][0][0][0]
        perc_sim_nn = compute_dists.compute(high_res[0,:,:], img_nearest[0,:,:])
        perc_sim_nn = perc_sim_nn.cpu().detach().numpy()[0][0][0][0]
        perc_sim_custom = compute_dists.compute(high_res[0, :, :], img_upsampled[0, :, :])
        perc_sim_custom = perc_sim_custom.cpu().detach().numpy()[0][0][0][0]

    else:
        perc_sim_bicubic = np.zeros(5)
        perc_sim_nn = np.zeros(5)
        for j in range(5):
            perc_sim_bicubic[j] = compute_dists.compute(high_res[j, :, :], img_bicubic[j, :, :])
            #perc_sim_bicubic[j] = perc_sim_bicubic.cpu().detach().numpy()[0][0][0][0]
            perc_sim_nn[j] = compute_dists.compute(high_res[j, :, :], img_nearest[j, :, :])
            #perc_sim_nn[j] = perc_sim_nn.cpu().detach().numpy()[0][0][0][0]

    #psnr_basic = compare_psnr(high_res_, low_for_psnr)
    if PLOT:
        if use_5_per == False:
            plot_image_grid([high_res, img_bicubic, img_nearest], directory, 'basic_compare_img_'+img_name_for_plot, 3, 12)
        else:
            for j in range(5):
                plot_image_grid([np.expand_dims(high_res[j,:,:], axis=0), np.expand_dims(img_bicubic[j,:,:], axis=0),
                                 np.expand_dims(img_nearest[j,:,:], axis=0)], directory,
                                'basic_compare_img_' + img_name_for_plot + 'frame_' + str(j), 3, 12)
        print ('PSNR bicubic: %.4f   PSNR nearest: %.4f' %  (psnr_bicubic, psnr_nn))
        print ('per_sim bicubic: %.4f   per_sim nearest: %.4f' %  (perc_sim_bicubic.mean(), perc_sim_nn.mean()))

    input_depth = 32
    INPUT = 'noise'
    pad = 'reflection'
    OPT_OVER = 'net'
    #KERNEL_TYPE = 'lanczos2'

    net_input = get_noise(input_depth, INPUT, (high_res.shape[-1], high_res.shape[-2])).type(dtype).detach()

    NET_TYPE = 'skip' # UNet, ResNet
    net = get_net(input_depth, 'skip', pad,
                  n_channels=high_res.shape[0],
                  skip_n33d=128,
                  skip_n33u=128,
                  skip_n11=4,
                  num_scales=5,
                  upsample_mode='bilinear').type(dtype)

    # Losses
    img_LR_var = np_to_torch(low_res).type(dtype)
    def closure():
        global i, net_input, psnr_history, psnr_history_short_HR, psnr_history_short_LR, orig_img_HR, per_sim_history

        if reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * reg_noise_std)

        out_HR = net(net_input)
        out_LR = custom_downsample(out_HR, factor, use_5_per)

        if loss_type == 'dip':
            total_loss = mse_fft(out_LR, img_LR_var)
            #total_loss = mse(out_LR, img_LR_var)
        elif loss_type == 'bp':
            total_loss = mse_bp(out_LR, img_LR_var.unsqueeze(0), factor)

        #print(total_loss)
        #print(tv_loss_try_abs(out_HR))
        if tv_weight > 0:
            mul_factor = 1
            total_loss = total_loss + tv_weight * tv_loss(out_HR, mul_factor).to(self.device)

        total_loss.backward()

        #high_res_ = np.squeeze(high_res, axis=0)
        #orig_img_HR = high_res_
        orig_img_HR = np.squeeze(high_res_, axis=0)
        orig_img_LR = low_res
        out_HR = out_HR.squeeze()
        out_LR = out_LR.squeeze()

        # History
        if i % 100 == 0:
            psnr_LR = compare_psnr(orig_img_LR, torch_to_np(out_LR), data_range=1)
            #psnr_HR = compare_psnr(orig_img_HR[:,pix_ignore:-pix_ignore, pix_ignore:-pix_ignore], torch_to_np(out_HR.unsqueeze(0))[:,pix_ignore:-pix_ignore, pix_ignore:-pix_ignore], data_range=1)
            psnr_HR = compare_psnr(orig_img_HR[pix_ignore:-pix_ignore, pix_ignore:-pix_ignore], torch_to_np(out_HR)[pix_ignore:-pix_ignore, pix_ignore:-pix_ignore], data_range=1)
            if use_5_per == False:
                perc_sim_HR = compute_dists.compute(orig_img_HR[pix_ignore:-pix_ignore, pix_ignore:-pix_ignore],
                                            torch_to_np(out_HR)[pix_ignore:-pix_ignore, pix_ignore:-pix_ignore])
            else:
                perc_sim_HR = np.zeros(5)
                for j in range(5):
                    perc_sim_HR[j] = compute_dists.compute(orig_img_HR[j, pix_ignore:-pix_ignore, pix_ignore:-pix_ignore],
                                                torch_to_np(out_HR)[j, pix_ignore:-pix_ignore, pix_ignore:-pix_ignore])
            psnr_history_short_HR.append([psnr_HR])
            psnr_history_short_LR.append([psnr_LR])
            per_sim_history.append([perc_sim_HR.mean()])

        if PLOT and i % 100 == 0:
            out_HR_np = torch_to_np(out_HR)
            if use_5_per == False:
                # plot_image_grid([np.expand_dims(high_res_, axis=0), img_nearest, img_bicubic, np.expand_dims(np.clip(out_HR_np, 0, 1), axis=0)],
                #                 directory, 'compare_img_'+img_name_for_plot, factor=13, nrow=4)
                plot_image_grid([high_res_, img_nearest, img_bicubic, np.expand_dims(np.clip(out_HR_np, 0, 1), axis=0)],
                                directory, 'compare_img_'+img_name_for_plot, factor=13, nrow=4)
                plt.imsave(directory + 'orig_'+img_name_for_plot + '.png', np.squeeze(high_res_), cmap='gray')
                plt.imsave(directory + 'nn_' + img_name_for_plot + '.png', np.squeeze(img_nearest), cmap='gray')
                plt.imsave(directory + 'bicubic_' + img_name_for_plot + '.png', np.squeeze(img_bicubic), cmap='gray')
                plt.imsave(directory + 'dip_' + img_name_for_plot + '.png', np.clip(out_HR_np, 0, 1), cmap='gray')
            else:
                for j in range(5):
                    plot_image_grid([np.expand_dims(high_res_[j,:,:], axis=0), np.expand_dims(img_bicubic[j,:,:], axis=0),
                                     np.expand_dims(np.clip(out_HR_np[j,:,:], 0, 1),
                                    axis=0)],directory, 'compare_img_' + img_name_for_plot + 'frame' + str(j), factor=13, nrow=3)
        if PLOT_PSNR and i % 100 == 0:
            print('Iteration %04d    PSNR %.3f   perc_sim %.3f' % (i, psnr_HR, perc_sim_HR.mean()), '\r')
        if i == num_iter - 1:
            out_HR_np = torch_to_np(out_HR)
            if use_5_per == False:
                # imwrite_multi_tiff([high_res_, np.squeeze(img_bicubic), np.clip(out_HR_np, 0, 1)],
                #                    directory + 'final_comparison_' + img_name_for_plot + '.tiff')
                imwrite_multi_tiff([np.squeeze(high_res_), np.clip(out_HR_np, 0, 1), np.squeeze(img_bicubic)],
                                   directory + 'final_comparison_' + img_name_for_plot + '.tiff')
            else:
                for j in range(5):
                    imwrite_multi_tiff([high_res_[j,:,:], np.clip(out_HR_np[j,:,:], 0, 1), np.squeeze(img_bicubic[j,:,:])],
                                       directory + 'final_comparison_' + img_name_for_plot + 'frame' + str(j) +'.tiff')
            #imwrite_multi_tiff([out_LR.detach().cpu().numpy(), img_LR_var.detach().cpu().numpy().squeeze()], directory + '1.tiff')
            #imwrite_multi_tiff([out_LR.detach().cpu().numpy(), img_LR_var.detach().cpu().numpy().squeeze()], directory + '1.tiff')

            # ### heat map ###
            # img_diff = (high_res_ - np.clip(out_HR_np, 0, 1)) ** 2
            # heat_map = ndimage.filters.gaussian_filter(img_diff, sigma=16)
            # max_val = np.max(heat_map)
            # min_val = np.min(heat_map)
            # norm_heat_map = (heat_map - min_val) / (max_val - min_val)
            # plt.imshow(high_res_)
            # plt.imshow(255 * norm_heat_map, alpha=0.5, cmap='viridis')
            # plt.axis('on')
            # plt.savefig(directory + 'heat_map_' + img_name_for_plot + '.jpg', bbox_inches='tight', pad_inches=0)

        i += 1

        return total_loss

    global psnr_history, psnr_history_short_HR, psnr_history_short_LR, orig_img_HR, per_sim_history
    psnr_history = []
    psnr_history_short_HR = []
    psnr_history_short_LR = []
    per_sim_history = []

    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()

    global i
    i = 0
    p = get_params(OPT_OVER, net, net_input)
    optimize(OPTIMIZER, p, closure, learning_rate, num_iter)

    out_HR_np = np.clip(torch_to_np(net(net_input)), 0, 1)
    #result_deep_prior = put_in_center(out_HR_np, imgs['orig_np'].shape[1:])

    return out_HR_np, orig_img_HR, net, net_input, psnr_history_short_HR, psnr_history_short_LR, \
           per_sim_history, psnr_bicubic, psnr_nn, psnr_custom, perc_sim_bicubic, perc_sim_nn, perc_sim_custom