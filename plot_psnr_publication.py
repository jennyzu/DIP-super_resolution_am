import numpy as np
import matplotlib.pyplot as plt

num_img = 8
add_det = '_scale5_5per'  # add more details regarding the run
#add_det = ''
resolution = ['HR', 'LR']

loss_type = 'dip'  # bp/dip/mixed
num_iter = 2000

directory = './results/' + loss_type + add_det + '/'

for k in range(1):
    f = open(directory + 'per_sim_history_%s.txt' % (loss_type), 'r')
    x = f.readlines()
    per_sim_per_iter = np.zeros(int(num_iter / 100))
    for i in range(len(x)):
        #single_per_sim = float(x[i][12:-(66-12-5)])
        single_per_sim = float(x[i].replace('[', '').replace(']', ''))
        j = i % (int(num_iter / 100))
        per_sim_per_iter[j] += single_per_sim
    per_sim_per_iter = per_sim_per_iter / num_img
    min_index = np.argmin(per_sim_per_iter)

    plt.plot(np.arange(0, num_iter, 100), per_sim_per_iter)
    plt.xlabel('iter #')
    plt.ylabel('PSNR')
    plt.title('Super Resolution - average per-sim (%s): %s_%s' % (resolution[k], add_det, loss_type))
    plt.text(.65, .8, 'min_avg_per_sim = %.3f' % per_sim_per_iter.min(), transform=plt.gca().transAxes)
    # plt.show()
    plt.savefig(directory + '/mean_persim%s_%s.png' % (add_det, loss_type))
    plt.close()
    f.close()

for k in range(1):
    f = open(directory + 'psnr_history_HR_%s.txt' % (loss_type), 'r')
    x = f.readlines()
    psnr_per_iter = np.zeros(int(num_iter / 100))
    for i in range(len(x)):
        #single_psnr = float(x[i][12:-(66-12-5)])
        single_psnr = float(x[i].replace('[', '').replace(']', ''))
        j = i % (int(num_iter / 100))
        psnr_per_iter[j] += single_psnr
    psnr_per_iter = psnr_per_iter / num_img

    plt.plot(np.arange(0, num_iter, 100), psnr_per_iter)
    plt.xlabel('iter #')
    plt.ylabel('PSNR')
    plt.title('Super Resolution - average psnr (%s): %s_%s' % (resolution[k], add_det, loss_type))
    plt.text(.65, .8, 'max_avg_psnr = %.3f' % psnr_per_iter.max(), transform=plt.gca().transAxes)
    plt.text(.65, .1, 'final_psnr = %.3f' % psnr_per_iter[min_index], transform=plt.gca().transAxes) #psnr in min value for per-sim
    # plt.show()
    plt.savefig(directory + '/mean_psnr%s_%s.png' % (add_det, loss_type))
    plt.close()
    f.close()


# ### figures per image
# for k in range(8):
#     f = open(directory + 'per_sim_history_%s.txt' % (loss_type), 'r')
#     x = f.readlines()
#     per_sim_per_iter = np.zeros(int(num_iter / 100))
#     for i in range(int(len(x)/(num_iter / 100))):
#         for j in range(int(num_iter / 100)):
#             #single_persim = float(x[j + i * int(num_iter / 100)].replace('[', '').replace(']', ''))
#             single_per_sim = float(x[j + i * int(num_iter / 100)][12:-(66-12-5)])
#             #j = i % (int(num_iter / 100))
#             per_sim_per_iter[j] = single_per_sim
#
#         plt.plot(np.arange(0, num_iter, 100), per_sim_per_iter)
#         plt.xlabel('iter #')
#         plt.ylabel('per_sim')
#         plt.title('SR - persim (HR): %s_%s_img #%d' % (add_det, loss_type, i))
#         plt.text(.65, .8, 'min_avg_persim = %.3f' % per_sim_per_iter.min(), transform=plt.gca().transAxes)
#         # plt.show()
#         plt.savefig(directory + '/mean_persim_HR_%s_%s_img%d.png' % (add_det, loss_type, i))
#         plt.close()
#
#     f.close()
