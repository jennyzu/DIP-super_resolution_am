import torch
import cv2
from PIL import Image
import numpy as np
import os

def tiff_imgs(crop_size, scale, use_5_per):

    if use_5_per == False:
        high_res = []
        low_res = []
        frames = [0, 1]

        dir = '/storage/SID/db/jenny_mdm/super_resolution/for_publication_db_all/'
        images = [i for i in os.listdir(dir)]
        #images = images[7:8]
        for image in images:
            img = Image.open(dir + image)
            for frame_ind in frames:
                try:
                    img.seek(frame_ind)
                    np_im = np.array(img)
                    if frame_ind == 0:  # lr image
                        img_l = (np_im - np_im.min()) / (np_im.max() - np_im.min())
                        #img_l = np_im / 255.0
                        low_res.append(img_l)
                    else:
                        np_im = cv2.resize(np_im, (crop_size * scale, crop_size * scale))
                        img_h = (np_im - np_im.min()) / (np_im.max() - np_im.min())
                        #img_h = np_im / 255.0
                        high_res.append(img_h)
                except:
                    print('Required Frame index doesn\'t exist ')

    elif use_5_per == True:
        high_res = []
        low_res = []
        frames = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        dir = '/storage/SID/db/jenny_mdm/super_resolution/for_publication_db_all/'
        images = [i for i in os.listdir(dir)]
        # images = images[7:8]
        for image in images:
            img = Image.open(dir + image)
            img_l_5 = np.zeros((5, crop_size, crop_size))
            img_h_5 = np.zeros((5, crop_size * scale, crop_size * scale))
            for frame_ind in frames:
                img.seek(frame_ind)
                np_im = np.array(img)
                if frame_ind < 5:  # lr image
                    #img_l = (np_im - np_im.min()) / (np_im.max() - np_im.min())
                    img_l = np_im / 255.0
                    img_l_5[frame_ind, :,:] = img_l
                else:
                    np_im = cv2.resize(np_im, (crop_size * scale, crop_size * scale))
                    #img_h = (np_im - np_im.min()) / (np_im.max() - np_im.min())
                    img_h = np_im / 255.0
                    img_h_5[frame_ind-5, :,:] = img_h
            low_res.append(img_l_5)
            high_res.append(img_h_5)

    return high_res, low_res
