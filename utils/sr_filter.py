import torch
import torch.nn.functional as nnf
import torchvision
import numpy as np
import cv2 as cv
import math
import cv2

def flip(x):
    return x.flip([2,3])

def fft2(s, n_x = [], n_y = []):
    h, w = s.shape[2], s.shape[3]
    if(n_x == []):
        n_x = w
    if(n_y == []):
        n_y = h
    s_pad = torch.nn.functional.pad(s, (0, n_x - w, 0 , n_y - h))

    return torch.rfft(s_pad, signal_ndim=2, normalized=True, onesided=False)

def mul_complex(t1, t2):
    ## Re{Z0 * Z1} = a0*a1 - b0*b1
    out_real = t1[:,:,:,:,0:1]*t2[:,:,:,:,0:1] - t1[:,:,:,:,1:2]*t2[:,:,:,:,1:2]
    ## Im{Z0 * Z1} = i*(a0*b1 + b0*a1)
    out_imag = t1[:,:,:,:,0:1]*t2[:,:,:,:,1:2] + t1[:,:,:,:,1:2]*t2[:,:,:,:,0:1]
    return torch.cat((out_real, out_imag), dim=4)

def abs2(x):
    out = torch.zeros_like(x)
    out[:,:,:,:,0] = x[:,:,:,:,0]**2 + x[:,:,:,:,1]**2
    return out

def bicubic_kernel_2D(x, y, a=-0.5):
    # get X
    abs_phase = np.abs(x)
    abs_phase3 = abs_phase**3
    abs_phase2 = abs_phase**2
    if abs_phase < 1:
        out_x = (a+2)*abs_phase3 - (a+3)*abs_phase2 + 1
    else:
        if abs_phase >= 1 and abs_phase < 2:
            out_x = a*abs_phase3 - 5*a*abs_phase2 + 8*a*abs_phase - 4*a
        else:
            out_x = 0
    # get Y
    abs_phase = np.abs(y)
    abs_phase3 = abs_phase**3
    abs_phase2 = abs_phase**2
    if abs_phase < 1:
        out_y = (a+2)*abs_phase3 - (a+3)*abs_phase2 + 1
    else:
        if abs_phase >= 1 and abs_phase < 2:
            out_y = a*abs_phase3 - 5*a*abs_phase2 + 8*a*abs_phase - 4*a
        else:
            out_y = 0

    return out_x*out_y

def get_bicubic(scale):
    size = 4 * scale + 2 + np.mod(scale, 2)
    is_even = not np.mod(scale, 2)
    grid_r = np.linspace(-(size//2) + 0.5*is_even,  size//2 - 0.5*is_even, size)
    r = np.zeros((size, size))
    for m in range(size):
        for n in range(size):
            r[m, n] = bicubic_kernel_2D(grid_r[n]/scale, grid_r[m]/scale)
    r = r/r.sum()

    return r

def get_gaus(scale, device):
    kernel_size = 16
    sigma = 1.6
    # scale: integer > 1
    x_coord = torch.arange(kernel_size).to(device)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * torch.exp(
        -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
        (2 * variance))
    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    return gaussian_kernel

def downsample_bicubic_2D(img, scale, device):
    # scale: integer > 1
    filter_supp = 4*scale + 2 + np.mod(scale, 2)
    is_even = 1 - np.mod(scale, 2)
    Filter = torch.zeros(1,1,filter_supp,filter_supp).float().to(device)
    grid = np.linspace(-(filter_supp//2) + 0.5*is_even, filter_supp//2 - 0.5*is_even, filter_supp)
    for n in range(filter_supp):
        for m in range(filter_supp):
            Filter[0, 0, m, n] = bicubic_kernel_2D(grid[n]/scale, grid[m]/scale)

    Filter = Filter/torch.sum(Filter)
    #pad = np.int((filter_supp - 1)/2)
    pad = np.int((filter_supp - scale)/2)
    img_padded = torch.nn.functional.pad(img, [pad, pad, pad, pad], mode='circular')
    img_out = torch.nn.functional.conv2d(img_padded, Filter, stride=(scale, scale))

    return img_out

def downsample_bicubic(img, scale, device):
    out = torch.zeros(img.shape[0], img.shape[1], img.shape[2]//scale, img.shape[3]//scale).to(device)
    out[:,:, :, :] = downsample_bicubic_2D(img[:, :, :, :], scale, device)
    return out

def downsample_custom(img, scale, layer, device):
    # scale: integer > 1
    kernel = np.load('kernels/kernel_' + layer + '.npy')
    kernel = torch.from_numpy(kernel).to(device)
    kernel = kernel.unsqueeze(0).unsqueeze(0)

    if scale == 3 or scale == 4:
        pad = np.int((kernel.shape[-1] - scale)/2)
        img_padded = torch.nn.functional.pad(img, [pad, pad, pad, pad], mode='circular')
        img_out = torch.nn.functional.conv2d(img_padded, kernel, stride=(scale, scale))
    # elif scale == 4:
    #     img = nnf.interpolate(img, size=(img.shape[-2]+scale,img.shape[-1] + scale), mode='bicubic')
    #     pad = np.int((kernel.shape[-1] - scale)/2)
    #     img_padded = torch.nn.functional.pad(img, [pad, pad, pad, pad], mode='circular')
    #     img_out = torch.nn.functional.conv2d(img_padded, kernel, stride=(scale, scale))
    elif scale ==2:
        img_out = torch.zeros(1,1,150,150).to(device)
        pad = np.int((kernel.shape[-1] - scale)/2)
        img_padded = torch.nn.functional.pad(img, [pad, pad, pad, pad], mode='circular')
        img_out [:,:,:-1,:-1] = torch.nn.functional.conv2d(img_padded, kernel, stride=(scale, scale))
        img_out [:,:,:,-1] = img_out [:,:,:,-2]
        img_out [:,:,-1,:] = img_out [:,:,-2,:]

    return img_out

def upsample_using_h(img, scale, layer, device):
    img = img.float().to(device)
    #h = np.load('kernels/kernel_' + layer + '.npy')
    h = np.load('kernels/bicubic.npy')
    h = torch.from_numpy(h).to(device)
    #h = h.unsqueeze(0).unsqueeze(0)

    flt_size = (h.shape[2], h.shape[3])
    #pad = 0
    #x_pad = torch.nn.functional.pad(img, [pad, pad, pad, pad], mode='circular')
    # img_up = torch.nn.functional.conv_transpose2d(x_pad, h, stride=scale,
    #         padding=(flt_size[0]//2 + np.int(np.ceil(scale/2)), flt_size[1]//2 + np.int(np.ceil(scale/2))))
    pad_size = int(0.5 * (scale*(img.shape[-1]-1) + flt_size[0] - img.shape[-1]*scale))
    img_up = torch.nn.functional.conv_transpose2d(img, h, stride=scale,
            padding=(pad_size, pad_size))
    img_up_torch = torch.nn.functional.interpolate(img, scale_factor=scale, mode='bicubic')

    return img_up, img_up_torch