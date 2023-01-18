import torch
import argparse
import os
from model import LeNet_Mnist, LeNet_Cifar
from utils import get_dataset, get_pogz_on_val
from torch.utils.data.dataloader import DataLoader
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# python get_pogz.py --dataset=fmnist --resume_path=./ckpt/lenet_mnist.pt --local_val_dataset_path=./data/FashionMNIST/

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='lenet', help='model name')
    parser.add_argument('--resume_path', type=str, help='load ckpt')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fmnist','cifar10'],
                        help='dataset')
    parser.add_argument('--local_val_dataset_path', type=str, default='./data/cifar10/',
                        help='load local dataset')
    parser.add_argument('--num_channels', type=int, default=3, help='num channels')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('Loading pretrained model ... ')

    if args.model == 'lenet':
        # lenet on mnist
        if args.dataset == 'mnist' or args.dataset == 'fmnist':
            #global_model = LeNet_Mnist(args=args)
            args.num_channels = 1
            client_model = LeNet_Mnist(args=args)
            #global_weights = global_model.state_dict()

        elif args.dataset == 'cifar10':
            #global_model = LeNet_Cifar(args=args)
            args.num_channels = 3
            client_model = LeNet_Cifar(args=args)
            #global_weights = global_model.state_dict()
            # client_model.load_state_dict(global_weights)

        else:
            exit('Error: unrecognized dataset')

    else:
        exit('Error: unrecognized model')

    client_model.load_state_dict(torch.load(args.resume_path))
    client_model.to(device)

    print('Loading local valid dataset ... ')
    _, _, valid_dataset = get_dataset(args)
    validloader = DataLoader(valid_dataset, batch_size=128, shuffle=False)


    # get pogz
    pogz_list = []
    pogz_list.append(get_pogz_on_val(client_model, validloader, args, device))


    if args.dataset == 'mnist' or args.dataset == 'fmnist':
        np.savez('./pogz/noise_' + args.model + '_' + args.dataset + '_pogz', conv1=np.array(pogz_list[0][0]),
                 conv2=np.array(pogz_list[0][1]), fc1=np.array(pogz_list[0][2]))
    elif args.dataset == 'cifar10':
        np.savez('./pogz/noise_' + args.model + '_' + args.dataset + '_pogz', conv1=np.array(pogz_list[0][0]), conv2=np.array(pogz_list[0][1]),
                 conv3=np.array(pogz_list[0][2]), fc1=np.array(pogz_list[0][3]))

    print('Done')
