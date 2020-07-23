import torch
import torchvision
import numpy as np
from scipy import interpolate
from scipy import fftpack
from scipy import integrate
from scipy import signal
import cv2 as cv

def init_weights(m):
    if type(m) == torch.nn.Linear or type(m) == torch.nn.Conv2d:
        torch.nn.init.normal_(m.weight, 0, 0.1)

def flip(x):
    return x.flip([2,3])

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

def get_bicubic(size, scale):
    is_even = not np.mod(scale, 2)
    grid_r = np.linspace(-(size//2) + 0.5*is_even,  size//2 - 0.5*is_even, size)
    r = np.zeros((size, size))
    for m in range(size):
        for n in range(size):
            r[m, n] = bicubic_kernel_2D(grid_r[n]/scale, grid_r[m]/scale)
    r = r/r.sum()

    return r

def downsample_bicubic_2D(I, scale, device):
    # scale: integer > 1
    filter_supp = 4*scale + 2 + np.mod(scale, 2)
    is_even = 1 - np.mod(scale, 2)
    Filter = torch.zeros(1,1,filter_supp,filter_supp).float().to(device)
    grid = np.linspace(-(filter_supp//2) + 0.5*is_even, filter_supp//2 - 0.5*is_even, filter_supp)
    for n in range(filter_supp):
        for m in range(filter_supp):
            Filter[0, 0, m, n] = bicubic_kernel_2D(grid[n]/scale, grid[m]/scale)

    Filter = Filter/torch.sum(Filter)
    pad = np.int((filter_supp - scale)/2)
    I_padded = torch.nn.functional.pad(I, [pad, pad, pad, pad], mode='circular')
    I_out = torch.nn.functional.conv2d(I_padded, Filter, stride=(scale, scale))

    return I_out

def downsample_bicubic(I, scale, device):
    out = torch.zeros(I.shape[0], I.shape[1], I.shape[2]//scale, I.shape[3]//scale).to(device)
    out[:,0:1, :, :] = downsample_bicubic_2D(I[:, 0:1, :, :], scale, device)
    out[:,1:2, :, :] = downsample_bicubic_2D(I[:, 1:2, :, :], scale, device)
    out[:,2:3, :, :] = downsample_bicubic_2D(I[:, 2:3, :, :], scale, device)
    return out

def filter_2D(I, h, device):
    pad_y = h.shape[0] // 2
    pad_x = h.shape[1] // 2
    Filter = torch.tensor(h).unsqueeze(0).unsqueeze(0).to(device).float()

    Filter = Filter/Filter.sum()
    
    out = torch.zeros_like(I).to(device)
    I_pad = torch.nn.functional.pad(I, (pad_x, pad_x, pad_y, pad_y), mode='circular')
    out[:, 0:1, :, :] = torch.nn.functional.conv2d(I_pad[:, 0:1, :, :], Filter, padding=(0, 0), stride=(1, 1))
    out[:, 1:2, :, :] = torch.nn.functional.conv2d(I_pad[:, 1:2, :, :], Filter, padding=(0, 0), stride=(1, 1))
    out[:, 2:3, :, :] = torch.nn.functional.conv2d(I_pad[:, 2:3, :, :], Filter, padding=(0, 0), stride=(1, 1))

    return out

def filter_2D_torch(I, Filter, device):
    pad_y = Filter.shape[2] // 2
    pad_x = Filter.shape[3] // 2

    Filter = Filter/Filter.sum()
    
    out = torch.zeros_like(I).to(device)
    I_pad = torch.nn.functional.pad(I, (pad_x, pad_x, pad_y, pad_y), mode='circular')
    out[:, 0:1, :, :] = torch.nn.functional.conv2d(I_pad[:, 0:1, :, :], Filter, padding=(0, 0), stride=(1, 1))
    if(I.shape[1] > 1):
        out[:, 1:2, :, :] = torch.nn.functional.conv2d(I_pad[:, 1:2, :, :], Filter, padding=(0, 0), stride=(1, 1))
        out[:, 2:3, :, :] = torch.nn.functional.conv2d(I_pad[:, 2:3, :, :], Filter, padding=(0, 0), stride=(1, 1))

    return out

class Upsampler(torch.nn.Module):
    def __init__(self, scale=4, kernel_size=16, in_channels=1, out_channels=1):
        super(Upsampler, self).__init__()
        self.pad = 1
        self.filter = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=scale,
            padding=kernel_size//2 + np.int(np.ceil(scale/2)), bias=False)

    def forward(self, x):
        x_pad = torch.nn.functional.pad(x, [self.pad, self.pad, self.pad, self.pad], mode='circular')
        return self.filter(x_pad)

def bicubic_up(img, scale, device):
    flt_size = 4*scale + np.mod(scale, 2)
    # upsampler = Upsampler(scale=scale, kernel_size=flt_size)
    # upsampler.to(device)
    is_even = 1 - np.mod(scale, 2)
    grid = np.linspace(-(flt_size//2) + 0.5*is_even, flt_size//2 - 0.5*is_even, flt_size)
    Filter = torch.zeros(1,1,flt_size, flt_size).to(device)
    for m in range(flt_size):
        for n in range(flt_size):
            Filter[0, 0, m, n] =  bicubic_kernel_2D(grid[n]/scale, grid[m]/scale)
    # Filter = Filter/Filter.sum()
    pad = 1
    x_pad = torch.nn.functional.pad(img, [pad, pad, pad, pad], mode='circular')
    img_up_torch = torch.nn.functional.interpolate(img, scale_factor=scale, mode='bicubic')
    img_up = torch.zeros_like(img_up_torch)
    for ch in range(3):
        img_up[:,ch:ch+1,:,:] = torch.nn.functional.conv_transpose2d(x_pad[:,ch:ch+1,:,:], Filter, stride=scale,
            padding=(flt_size//2 + np.int(np.ceil(scale/2)), flt_size//2 + np.int(np.ceil(scale/2))))

    return img_up#, img_up_torch
    
def fft2(s, n_x = [], n_y = []):
    h, w = s.shape[2], s.shape[3]
    if(n_x == []):
        n_x = w
    if(n_y == []):
        n_y = h 
    s_pad = torch.nn.functional.pad(s, (0, n_x - w, 0 , n_y - h))

    return torch.rfft(s_pad, signal_ndim=2, normalized=False, onesided=False)

def mul_complex(t1, t2):
    ## Re{Z0 * Z1} = a0*a1 - b0*b1
    out_real = t1[:,:,:,:,0:1]*t2[:,:,:,:,0:1] - t1[:,:,:,:,1:2]*t2[:,:,:,:,1:2]
    ## Im{Z0 * Z1} = i*(a0*b1 + b0*a1)
    out_imag = t1[:,:,:,:,0:1]*t2[:,:,:,:,1:2] + t1[:,:,:,:,1:2]*t2[:,:,:,:,0:1]
    return torch.cat((out_real, out_imag), dim=4)

def conj(x):
    out = x
    out[:,:,:,:,1] = -out[:,:,:,:,1]
    return out 

def abs2(x):
    out = torch.zeros_like(x)
    out[:,:,:,:,0] = x[:,:,:,:,0]**2 + x[:,:,:,:,1]**2
    return out

def correct_img_torch(x_s, scale, r, s, device, for_dag = True, eps = 1e-9):
    conv_shape = (s.shape[2] + r.shape[2] - 1, s.shape[3] + r.shape[3] - 1)
    S = fft2(s, conv_shape[1], conv_shape[0])
    R = fft2(r, conv_shape[1], conv_shape[0])
    Q_unscaled = mul_complex(R, S)
    q_unscaled = torch.irfft(Q_unscaled, signal_ndim=2, normalized=True, onesided=False)
    q = q_unscaled[:,:,np.mod(q_unscaled.shape[2], scale)::scale, np.mod(q_unscaled.shape[3], scale)::scale]
    Q = torch.rfft(q, signal_ndim=2, normalized=True, onesided=False)

    Q_star = conj(Q)
    abs2_Q = abs2(Q)
    H = torch.cat( (Q_star[:,:,:,:,0:1]/(abs2_Q[:,:,:,:,0:1] + eps), Q_star[:,:,:,:,1:2]/(abs2_Q[:,:,:,:,0:1] + eps)), dim=4)
    h = torch.irfft(H, signal_ndim=2, normalized=True, onesided=False)
    h = h/h.sum()
    h = roll_x(h, -1)
    h = roll_y(h, -1)

    x_h = filter_2D_torch(x_s, flip(h), device)

    if(for_dag):
        x_h = bicubic_up(x_h, scale, device)
        x_h = downsample_bicubic(x_h, scale, device)

    return x_h, h
