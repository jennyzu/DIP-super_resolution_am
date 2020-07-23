import torch
import cv2
from PIL import Image
import numpy as np
import os

crop_size = 90

def tiff_imgs(img_path, scale, use_5_per):
    if use_5_per == False:
        img = Image.open(img_path)
        list_of_images = []
        frames = [0, 1]

        for frame_ind in frames:
            img.seek(frame_ind)
            np_im = np.array(img)
            if frame_ind == 0:  # lr image
                img_l = (np_im - np_im.min()) / (np_im.max() - np_im.min())
                #img_l = np_im / 255.0
                list_of_images.append(img_l)
            else:
                np_im = cv2.resize(np_im, (crop_size * scale, crop_size * scale))
                img_h = (np_im - np_im.min()) / (np_im.max() - np_im.min())
                #img_h = np_im / 255.0
                list_of_images.append(img_h)

    elif use_5_per == True:
        img = Image.open(img_path)
        list_of_images = []
        frames = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        img_l_5 = np.zeros((5, crop_size, crop_size))
        img_h_5 = np.zeros((5, crop_size * scale, crop_size * scale))
        for frame_ind in frames:
            img.seek(frame_ind)
            np_im = np.array(img)
            if frame_ind < 5:  # lr image
                img_l = (np_im - np_im.min()) / (np_im.max() - np_im.min())
                #img_l = np_im / 255.0
                img_l_5[frame_ind, :, :] = img_l
            else:
                np_im = cv2.resize(np_im, (crop_size * scale, crop_size * scale))
                img_h = (np_im - np_im.min()) / (np_im.max() - np_im.min())
                #img_h = np_im / 255.0
                img_h_5[frame_ind-5, :, :] = img_h
        list_of_images.append(img_l_5)
        list_of_images.append(img_h_5)

    return list_of_images