from torchvision import datasets, transforms
import os
from torch.utils.data import Dataset
import torch

def get_pogz_on_val(model, data, args, device):
    model.eval()
    model.reset()
    if args.model == 'lenet':
        for batch_idx, (images, labels) in enumerate(data):
            images, labels = images.to(device), labels.to(device)
            model.activityNum_forward(images)
    else:
        exit('Error: unrecognized model')
    # data_num, noise_type
    pogz_list = model.get_pogz(len(data.dataset))

    return pogz_list


def get_dataset(args):
    """ Returns train and test datasets.
    """
    if args.dataset == 'cifar10':

        train_transform = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914672374725342, 0.4822617471218109, 0.4467701315879822),
                                 (0.24703224003314972, 0.24348513782024384, 0.26158785820007324)),
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914672374725342, 0.4822617471218109, 0.4467701315879822),
                                 (0.24703224003314972, 0.24348513782024384, 0.26158785820007324)),
        ])
        train_dataset = datasets.ImageFolder(root=args.local_val_dataset_path + '/train/', transform=train_transform)
        test_dataset = datasets.ImageFolder(root=args.local_val_dataset_path + '/test/', transform=test_transform)

    elif args.dataset == 'mnist':
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = datasets.MNIST(args.local_val_dataset_path, train=True, download=True,
                                       transform=train_transform)

        test_dataset = datasets.MNIST(args.local_val_dataset_path, train=False, download=True,
                                      transform=test_transform)

    elif args.dataset == 'fmnist':
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = datasets.FashionMNIST(args.local_val_dataset_path, train=True, download=True,
                                       transform=train_transform)

        test_dataset = datasets.FashionMNIST(args.local_val_dataset_path, train=False, download=True,
                                      transform=test_transform)
    else:
        NotImplementedError()

    if os.path.exists(args.local_val_dataset_path+'/val/'):
        val_dataset = datasets.ImageFolder(root=args.local_val_dataset_path + '/val/', transform=test_transform)

    else:
        idxs_train_all = [i for i in range(len(train_dataset))]
        idxs_val = idxs_train_all[(int(0.9 * len(idxs_train_all))):]
        idxs_train = idxs_train_all[:int(0.9 * len(idxs_train_all))]
        # print(idxs_val)
        train_dataset_ori = train_dataset

        train_dataset = DatasetSplit(train_dataset_ori, idxs_train)
        val_dataset = DatasetSplit(train_dataset_ori, idxs_val)

    return train_dataset, test_dataset, val_dataset


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)