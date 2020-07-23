import torch
import load_images_kernel_publication
import cv2
import numpy as np
import imageio
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

from typing import Union
import os
from PIL import Image

scale = 5
crop_size = 90
kernel_npy = 'kernels/kernel_pad.npy'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

def blur(img, flt, scale, device):
    pad_size = int(flt.shape[-1] / 2)
    img_padded = torch.nn.functional.pad(img, [pad_size, pad_size, pad_size, pad_size])#, mode='circular')
    #img_out = torch.zeros_like(img).to(device)
    img_out = torch.nn.functional.conv2d(img_padded, flt, stride=scale)

    return img_out

# low resolution image
dataset = load_images_kernel_publication.tiff_imgs(scale, crop_size)
#img_l = create_lr_img.create_uniform_lr()
#img_l = cv2.convertScaleAbs(img_l, alpha=(255.0/65535.0))
img_l = dataset[1][2]
img_l = torch.from_numpy(img_l).to(device)
img_l = img_l.unsqueeze(0).unsqueeze(0)
img_l = img_l.float()

#high resolution image
#img_h = cv2.convertScaleAbs(dataset[0][0], alpha=(255.0/65535.0))
img_h = dataset[0][2]
img_h = torch.from_numpy(img_h).to(device)
img_h = img_h.unsqueeze(0).unsqueeze(0)
img_h = img_h.float()

# my kernel
my_kernel = np.load(kernel_npy)
my_kernel = torch.from_numpy(my_kernel).to(device)
my_kernel = my_kernel.unsqueeze(0).unsqueeze(0)
img_l_my = blur(img_h, my_kernel, scale, device)
orig_low_image = img_l.squeeze().cpu().numpy()
my_img = img_l_my.squeeze().cpu().numpy()
#my_img_imwrite = (my_img - my_img.min()) / (my_img.max() - my_img.min())
psnr_my = compare_psnr(orig_low_image, my_img, data_range=1)
imageio.imwrite('images_check_kernel/orig.png', (255*orig_low_image).astype(np.uint8))
imageio.imwrite('images_check_kernel/my_img.png', (255*my_img).astype(np.uint8))#.astype(np.uint8))
imwrite_multi_tiff([orig_low_image, my_img], 'images_check_kernel/try.tiff')
imwrite_multi_tiff([cv2.resize(orig_low_image, (450,450)), cv2.resize(my_img, (450,450))], 'images_check_kernel/try1.tiff')

# basic kernel
basic_filter = torch.ones(1, 1, 15, 15).to(device)
basic_filter = basic_filter / basic_filter.sum()
img_l_basic = blur(img_h, basic_filter, scale, device)
basic_image = img_l_basic.squeeze().cpu().numpy()
psnr_basic = compare_psnr(orig_low_image, basic_image, data_range=1)
imageio.imwrite('images_check_kernel/uniform_flt.png', (255*basic_image).astype(np.uint8))#.astype(np.uint8))

a=1


