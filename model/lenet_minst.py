from torch import nn
import numpy as np
import torch


class LeNet_Mnist(nn.Module):
    def __init__(self, args):
        super(LeNet_Mnist, self).__init__()
        self.conv1 = self.conv1 = nn.Conv2d(args.num_channels, 12, kernel_size=5, padding=5//2, stride=2)
        self.act1 = nn.Sigmoid()
        self.conv2 = nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=2)
        self.act2 = nn.Sigmoid()
        self.fc1 = nn.Linear(12 * 7 * 7, 10)

        self.conv1_output_shape = [12, 14, 14]
        self.conv2_output_shape = [12, 7, 7]

    def forward(self, x):
        x = self.conv1(x)
        self.conv1_output_shape = x.shape
        x = self.act1(x)

        x = self.conv2(x)
        self.conv2_output_shape = x.shape
        x = self.act2(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return x

    def reset(self):
        self.layer_conv1_output = np.zeros(self.conv1_output_shape)
        self.layer_conv2_output = np.zeros(self.conv2_output_shape)

        self.layer_fc1_input_lip = np.zeros(self.fc1.in_features)
        self.layer_fc1_output = np.zeros(self.fc1.out_features)
        self.layer_fc1_output_lip = np.zeros(self.fc1.out_features)

        self.POGZ_conv1 = 0
        self.POGZ_conv2 = 0
        self.POGZ_fc1 = 0

    def activityNum_forward(self, x):
        x = self.conv1(x)
        have_sum = np.sum(np.where(x.cpu().detach().numpy() > 0, 1, 0), axis=0)
        self.layer_conv1_output = have_sum + self.layer_conv1_output
        x = self.act1(x)

        x = self.conv2(x)
        have_sum = np.sum(np.where(x.cpu().detach().numpy() > 0, 1, 0), axis=0)
        self.layer_conv2_output = have_sum + self.layer_conv2_output
        x = self.act2(x)

        x = x.view(x.size(0), -1)

        self.layer_fc1_input_lip = np.mean(x.cpu().detach().numpy(), axis=0) + self.layer_fc1_input_lip
        x = self.fc1(x)
        self.layer_fc1_output_lip = np.mean(x.cpu().detach().numpy(), axis=0) + self.layer_fc1_output_lip
        have_sum = np.sum(np.where(x.cpu().detach().numpy() > 0, 1, 0), axis=0)
        self.layer_fc1_output = have_sum + self.layer_fc1_output

        return x

    def get_pogz(self, data_num):
        # cal POGZ
        self.POGZ_conv1 = np.sum(self.layer_conv1_output) / (data_num * self.layer_conv1_output.size)
        self.POGZ_conv2 = np.sum(self.layer_conv2_output) / (data_num * self.layer_conv2_output.size)
        self.POGZ_fc1 = np.sum(self.layer_fc1_output) / (data_num * self.layer_fc1_output.size)

        POGZ_sum = self.POGZ_conv1 + self.POGZ_conv2 + self.POGZ_fc1

        if POGZ_sum != 0:
            self.POGZ_conv1 = self.POGZ_conv1 / POGZ_sum
            self.POGZ_conv2 = self.POGZ_conv2 / POGZ_sum
            self.POGZ_fc1 = self.POGZ_fc1/POGZ_sum
        else:
            self.POGZ_conv1 = 1/3
            self.POGZ_conv2 = 1/3
            self.POGZ_fc1 = 1/3

        # Allocate epsilon
        POGZ_list = [self.POGZ_conv1, self.POGZ_conv2, self.POGZ_fc1]

        return POGZ_list

    def add_noise(self, POGZ_list, weights, device, sensitivity, epsilon):
        pogz_sum = POGZ_list['conv1'] + POGZ_list['conv2'] + POGZ_list['fc1']
        epsilon_conv1 = (POGZ_list['conv1'] / pogz_sum) * epsilon
        epsilon_conv2 = (POGZ_list['conv2'] / pogz_sum) * epsilon
        epsilon_fc1 = (POGZ_list['fc1'] / pogz_sum) * epsilon

        scale_conv1 = sensitivity / epsilon_conv1
        scale_conv2 = sensitivity / epsilon_conv2
        scale_fc1 = sensitivity / epsilon_fc1

        weights['conv1.weight'] = weights['conv1.weight'] + torch.from_numpy(np.random.laplace(loc=0, scale=scale_conv1, size=1)).to(device).float()
        weights['conv2.weight'] = weights['conv2.weight'] + torch.from_numpy(np.random.laplace(loc=0, scale=scale_conv2, size=1)).to(device).float()
        weights['fc1.weight'] = weights['fc1.weight'] + torch.from_numpy(np.random.laplace(loc=0, scale=scale_fc1, size=1)).to(device).float()

        return weights

