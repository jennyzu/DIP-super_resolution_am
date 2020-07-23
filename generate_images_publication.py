import json
import matplotlib.image as mpimg
import numpy as np
from scipy import interpolate
import cv2

from typing import Union
import os
from PIL import Image

use_5_per = True
scale = 4 / 0.75
img_size = 480
frame = 0
folder = 'Res' #SEM_GA/Stig/Res
raw_url = '/storage/SID/db/jenny_mdm/for_publication_raw/'
save_url = '/storage/SID/db/jenny_mdm/super_resolution/for_publication_db_all/'
json_file_name = r'/storage/SID/db/jenny_mdm/for_publication_raw/Res/info_Res.json'
with open(json_file_name, 'r') as readfile:
    Samples = json.load(readfile)

def imread_multi_tiff(filename, frames=None):
    # input : filename is the ".tiff" file name
    #
    _, file_extension = os.path.splitext(filename)

    assert file_extension.lower() == '.tiff' or file_extension.lower() == '.tif', 'error - file extention must be .tiff'
    if frames is None:
        max_frames = 0
        try:
            img = Image.open(filename)
            while True:
                img.seek(max_frames)
                max_frames += 1
        except:
            frames = range(max_frames)
    if not type(frames) is list:
        frames = [frames]

    list_of_images = []
    img = Image.open(filename)
    for frame_ind in frames:
        try:
            img.seek(frame_ind)
            # Convert to numpy array
            np_im = np.array(img)
            list_of_images.append(np_im)
        except EOFError:
            print('Wrong frame index : ' + str(frame_ind))
        except:
            print('Required Frame index doesn\'t exist ')
    return list_of_images

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

samples_list = [Samples[n] for n in Samples if Samples[n]['best_scale'] < 10]

for i in range(len(samples_list)):
    sample = samples_list[i]
    Scale = sample['best_scale_fixed']
    # Read the images
    #im_Class = imread_multi_tiff(sample['HR_linux'], [frame])[0]
    im_Class = imread_multi_tiff(sample['HR_linux'], [0, 1, 2, 3, 4]) ##
    #im_Class = im_Class / 255 # original images are uint16
    #im_Defect = imread_multi_tiff(sample['LR_linux'], [frame])[0]
    im_Defect = imread_multi_tiff(sample['LR_linux'], [0, 1, 2, 3, 4]) ##
    #im_Defect = im_Defect / 255 # original images are uint16

    if use_5_per == False:
        Target_size = im_Class.shape
        Shift_X = -(im_Defect.shape[1] / 2) / Scale + Target_size[1] / 2 - sample["X_offset_Main"] + 1.5
        Shift_Y = -(im_Defect.shape[0] / 2) / Scale + Target_size[0] / 2 - sample["Y_offset_Main"] + 1.5
        M = np.float32([[1 / Scale, 0, Shift_X], [0, 1 / Scale, Shift_Y]])

        #Getting Hi Res image coordinates on Low res image
        Class_coordinates = np.array([[0, 0], np.array(im_Class.shape)])
        new_coordinates = (Class_coordinates - M[:, 2]).dot(np.linalg.inv(M[:2, :2]))
        ul = new_coordinates[0, :] #upper left
        lr = new_coordinates[1, :] #lower right

        target_size = im_Defect.shape[0]
        target_size_new = int(lr[0] - ul[0])
        x = np.linspace(0, target_size - 1, target_size)
        y = np.linspace(0, target_size - 1, target_size)
        x_new = np.linspace(ul[1] + 0.5, lr[1] - 0.5, num=target_size_new)
        y_new = np.linspace(ul[0] + 0.5, lr[0] - 0.5, num=target_size_new)
        f = interpolate.interp2d(x, y, im_Defect, kind='cubic')
        im_lr = f(y_new, x_new)

        for ind, c in enumerate(sample['HR_linux']):
            if c.isdigit():
                break
        imwrite_multi_tiff([im_lr, im_Class], save_url + folder + '_' + sample['HR_linux'][ind:ind+5] + '_.tiff')

    elif use_5_per == True:
        Target_size = im_Class[0].shape
        Shift_X = -(im_Defect[0].shape[1] / 2) / Scale + Target_size[1] / 2 - sample["X_offset_Main"] + 1.5
        Shift_Y = -(im_Defect[0].shape[0] / 2) / Scale + Target_size[0] / 2 - sample["Y_offset_Main"] + 1.5
        M = np.float32([[1 / Scale, 0, Shift_X], [0, 1 / Scale, Shift_Y]])

        Class_coordinates = np.array([[0, 0], np.array(im_Class[0].shape)])
        new_coordinates = (Class_coordinates - M[:, 2]).dot(np.linalg.inv(M[:2, :2]))
        ul = new_coordinates[0, :] #upper left
        lr = new_coordinates[1, :] #lower right

        target_size = im_Defect[0].shape[0]
        target_size_new = int(lr[0] - ul[0])
        x = np.linspace(0, target_size-1, target_size)
        y = np.linspace(0, target_size-1, target_size)
        x_new = np.linspace(ul[1] + 0.5, lr[1] - 0.5, num=target_size_new)
        y_new = np.linspace(ul[0] + 0.5, lr[0] - 0.5, num=target_size_new)
        im_lr = [None]*5
        for channel in range(5):
            #f = interpolate.interp2d(x, y, im_Defect[:,:,channel], kind='cubic')
            f = interpolate.interp2d(x, y, im_Defect[channel], kind='cubic')
            im_lr[channel] = f(y_new,x_new)

        for ind, c in enumerate(sample['HR_linux']):
            if c.isdigit():
                break
        imwrite_multi_tiff(im_lr + im_Class, save_url + folder + '_' + sample['HR_linux'][ind:ind + 5] + '_.tiff')


    #imwrite_multi_tiff([cv2.resize(im_lr, (450, 450)), cv2.resize(im_Class, (450, 450))], save_url + folder + '_' + sample['HR_linux'][ind:ind+5] + '_try.tiff')

    #im_lr_resized = cv2.resize(im_lr, im_Class.shape)
    #imwrite_multi_tiff([im_lr_resized, im_Class], save_url + 'for_MDM_sample_ID_' + sample['Layer'] + '_' + str(sample['xls_ID']) + '_toggle.tiff')

