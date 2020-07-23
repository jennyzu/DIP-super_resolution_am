import torch
import torch.utils.data
import numpy as np
import utils
import os
from torch import nn
import load_images_kernel_publication
import cv2

crop_size = 90
scale = 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_5_per = True

class kernel_estimator(nn.Module):
    def __init__(self, scale):
        super(kernel_estimator, self).__init__()
        if scale % 2 == 0: #even scale->even filter size
            self.conv = nn.Conv2d(1, 1, kernel_size=16, stride=scale, padding=8)
        else:
            ker_s = 25
            self.conv = nn.Conv2d(1, 1, kernel_size=ker_s, stride=scale, padding=np.int((ker_s-scale)//2))
            #bicubic_kernel = torch.from_numpy(np.load('kernels/bicubic.npy')).to(device)
            #self.conv.weight.data = bicubic_kernel
        #print(self.conv.weight.data)

        '''
        for param in model.parameters():
            if param.data.shape[-1] == 25:
                kernel = param.data
                kernel = kernel.squeeze()

        data = kernel.cpu().numpy()
        np.save('kernels/kernel', data)
        '''

    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.conv(x)

        return x

lr = 1e-4
batch_size = 1
Epochs = 800000
print_every = 100

high_res, low_res = load_images_kernel_publication.tiff_imgs(crop_size, scale, use_5_per)
low_res = torch.Tensor(low_res)
high_res = torch.Tensor(high_res)
dataset = torch.utils.data.TensorDataset(high_res, low_res)

train_loader = torch.utils.data.DataLoader(
              dataset, batch_size=batch_size, shuffle=False)

model = kernel_estimator(scale).to(device)
model.train()
objective = torch.nn.L1Loss()
#objective = torch.nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-3, lr=lr)

iteration = 0
for epoch in range(Epochs):
    acc_loss = 0
    for batch_num, (input, target) in enumerate(train_loader):
        if use_5_per == True:
            x_h = input[0].unsqueeze(0)
            x_h = x_h.view(-1, 1 , x_h.shape[-1], x_h.shape[-1])
        elif use_5_per == False:
            x_h = input[0].unsqueeze(0).unsqueeze(0)
        x_h = x_h.float()
        x_h = x_h.to(device)
        optimizer.zero_grad()
        x_h_out = model(x_h)

        img_l = target[0].unsqueeze(0).unsqueeze(0)
        img_l = img_l.float().to(device)
        loss = objective(x_h_out, img_l)
        acc_loss += loss.item()
        loss.backward()
        optimizer.step()

    if (iteration % print_every == 0):
        print('Epoch %3d, Iteration %5d, Average Loss %.10f' % (epoch, iteration, acc_loss / (print_every*36)))
        acc_loss = 0

    iteration += 1

if (torch.save(model.state_dict(), './model.pth')):
    print('saved model')






