import torch
import time
import argparse
import os
from model import LeNet_Mnist, LeNet_Cifar
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = "0"


if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='lenet', help='model name')
    parser.add_argument('--resume_path', type=str, help='load ckpt')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fmnist','cifar10'],
                        help='dataset')
    parser.add_argument('--num_channels', type=int, default=3, help='num channels')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('Loading updated client model ... ')
    if args.model == 'lenet':
        if args.dataset == 'mnist' or args.dataset == 'fmnist':
            args.num_channels = 1
            updated_client_model = LeNet_Mnist(args=args)
        elif args.dataset == 'cifar10':
            args.num_channels = 3
            updated_client_model = LeNet_Cifar(args=args)
        else:
            exit('Error: unrecognized dataset')
    else:
        exit('Error: unrecognized model')

    updated_client_model.load_state_dict(torch.load(args.resume_path))
    updated_client_model.to(device)

    print('Loading pogz list ... ')
    pogz_list = np.load('./pogz/noise_' + args.model + '_' + args.dataset + '_pogz.npz')

    print('Adding noise ... ')
    alpha = 1.0
    lr = 0.1
    sensitivity = 2*alpha*lr
    epsilon = 10.0
    # self, POGZ_list, weights, device, sensitivity, epsilon
    noised_weights = updated_client_model.add_noise(pogz_list, updated_client_model.state_dict(), device, sensitivity, epsilon)
    updated_client_model.load_state_dict(noised_weights)
    print('Done')
