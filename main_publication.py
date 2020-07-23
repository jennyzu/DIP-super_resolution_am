import torch
import cv2 as cv
import numpy as np
import random
from models.unet import UNet
import csv
import os
import super_resolution_publication
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from PerceptualSimilarity import compute_dists


# random seed
seed_num = 123
torch.manual_seed(seed_num)
torch.cuda.manual_seed(seed_num)
np.random.seed(seed_num)
random.seed(seed_num)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True

# parameters to set
use_5_per = False
scale = 5
loss_type = 'dip' #bp/dip/mixed
add_det = '_scale5_extra_images' #add more details regarding the run
#add_det = ''

#directory = './results/super_resolution/' + 'try/'
directory = './results/' + loss_type + add_det + '/'
print(directory)
if not os.path.exists(directory):
    os.mkdir(directory)
imgs_dir = '/storage/SID/db/jenny_mdm/super_resolution/for_publication_db_1/'
files = [f for f in os.listdir(imgs_dir) if os.path.isfile(os.path.join(imgs_dir, f))]
GT_imgs = [f for f in files if '.tiff' in f]
final_psnr_file = open(directory + 'final_psnr_perc_log_%s.csv' % loss_type, 'a')
PSNR_fileWriter = csv.writer(final_psnr_file)
basic_compare_file = open(directory + 'bicubic_nn_log_%s.csv' % loss_type, 'a')
basic_fileWriter = csv.writer(basic_compare_file)

for img in GT_imgs:
    #if img == 'for_MDM_sample_ID_Layer_23.5_7628.tiff':
    pix_ignore = 6 + scale # ignore pixels from border when calc psnr (6 + factor)
    I_DIP, I, network, z, psnr_HR, psnr_LR, per_sim, psnr_bi, psnr_nn, psnr_custom, perc_bi, perc_nn, perc_custom = super_resolution_publication.dip_sr(
        img, loss_type, directory, pix_ignore, scale, imgs_dir, use_5_per)
    I_DIP = I_DIP.squeeze()
    final_psnr = compare_psnr(I[pix_ignore:-pix_ignore, pix_ignore:-pix_ignore], I_DIP[pix_ignore:-pix_ignore, pix_ignore:-pix_ignore])
    final_perc = compute_dists.compute(I[pix_ignore:-pix_ignore, pix_ignore:-pix_ignore], I_DIP[pix_ignore:-pix_ignore, pix_ignore:-pix_ignore])
    print('psnr = %.4f' % (final_psnr))

    row_str = ['%f %f' % (final_psnr, final_perc)]
    PSNR_fileWriter.writerow(row_str)
    row_str1 = ['%f %f %f %f %f %f' % (psnr_bi, psnr_nn, psnr_custom, perc_bi, perc_nn, perc_custom)]
    basic_fileWriter.writerow(row_str1)

    ### for deciding number of iterations based on average psnr
    with open(directory + 'psnr_history_HR_%s.txt' % loss_type, 'a') as f:
        for item in psnr_HR:
            f.write("%s\n" % item)
    with open(directory + 'psnr_history_LR_%s.txt' % loss_type, 'a') as f:
        for item in psnr_LR:
            f.write("%s\n" % item)
    with open(directory + 'per_sim_history_%s.txt' % loss_type, 'a') as f:
        for item in per_sim:
            f.write("%s\n" % item)

final_psnr_file.close()
basic_compare_file.close()