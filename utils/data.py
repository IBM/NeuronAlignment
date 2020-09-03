import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from utils import tinyimagenet200_data


class Transforms:
    class MNIST:
        class VGG:
            train = transforms.Compose([
                transforms.ToTensor(),
            ])

            test = transforms.Compose([
                transforms.ToTensor(),
            ])
            
            test_clean = test     
        TinyTen = VGG
        ResNet32 = VGG
        TinySix = VGG
        ConvFC = VGG
        Inception = VGG

    class CIFAR10:

        class Adversarial:
            train = transforms.Compose([
                transforms.RandomAffine(degrees=15),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
            ])

            test = transforms.Compose([
                transforms.ToTensor(),
            ])

            test_clean = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            ])

        class VGG:
            train = transforms.Compose([
                transforms.RandomAffine(degrees=15),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            ])

            test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            ])
           
            test_clean = test

        class ResNet32:
            train = transforms.Compose([
                transforms.RandomAffine(degrees=15),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            ])

            test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            ])

            test_clean = test

        class TinyTen:
            train = transforms.Compose([
                transforms.RandomAffine(degrees=15),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            ])

            test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            ])

            test_clean = test

        GoogLeNet = TinyTen
        TinyThree = TinyTen

    class TINYIMAGENET200:

        class TinyTen:
            train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(64, padding=8),
                transforms.RandomAffine(degrees=30, scale=(0.8, 1.2)),
                transforms.ColorJitter(contrast=0.25, saturation=0.25),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            test_clean = test

        ResNet32 = TinyTen
        VGG = TinyTen
        TinySix = TinyTen

        class GoogLeNet:
            train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Resize(32, interpolation=2), 
                transforms.RandomCrop(32, padding=4),
                transforms.RandomAffine(degrees=30, scale=(0.8, 1.2)),
                transforms.ColorJitter(contrast=0.25, saturation=0.25),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            test = transforms.Compose([
                transforms.Resize(32, interpolation=2), 
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            test_clean = test

        class Adversarial:
            train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(64, padding=8),
                transforms.RandomAffine(degrees=30, scale=(0.8, 1.2)),
                transforms.ColorJitter(contrast=0.25, saturation=0.25),
                transforms.ToTensor()
            ])

            test = transforms.Compose([
                transforms.ToTensor()
            ])

            test_clean = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            ])
        
        class Adversarial_GoogLeNet:
            train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Resize(32, interpolation=2), 
                transforms.RandomCrop(32, padding=4),
                transforms.RandomAffine(degrees=30, scale=(0.8, 1.2)),
                transforms.ColorJitter(contrast=0.25, saturation=0.25),
                transforms.ToTensor()
            ])

            test = transforms.Compose([
                transforms.Resize(32, interpolation=2), 
                transforms.ToTensor()
            ])

            test_clean = transforms.Compose([
                transforms.Resize(32, interpolation=2), 
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            ])

    CIFAR100 = CIFAR10
    CIFAR100_HALF = CIFAR100
    SVHN = MNIST
    ImageNet = CIFAR10


def loaders(dataset, path, batch_size, num_workers, transform_name, use_test=False,
            shuffle_train=True, test_batch_size=256):
    path_old = path
    path = os.path.join(path, dataset.lower())
    dataset = dataset.replace('-', '')
    if dataset == 'IMAGENET':
        dataset = 'ImageNet'
    transform = getattr(getattr(Transforms, dataset), transform_name)
    if dataset == 'CIFAR10' or dataset == 'CIFAR100' or dataset == 'ImageNet':
        ds = getattr(datasets, dataset)
        train_set = ds(path, train=True, download=True, transform=transform.train)
        align_set = ds(path, train=True, download=True, transform=transform.test)
        if use_test:
            test_set = ds(path, train=False, download=True, transform=transform.test)
            test_clean_set = ds(path, train=False, download=True, transform=transform.test_clean)
        else:
            train_set.train_data = train_set.train_data[:-5000]
            train_set.train_labels = train_set.train_labels[:-5000]

            test_set = ds(path, train=True, download=True, transform=transform.test)
            test_set.train = False
            test_set.test_data = test_set.train_data[-5000:]
            test_set.test_labels = test_set.train_labels[-5000:]
            delattr(test_set, 'train_data')
            delattr(test_set, 'train_labels')
            
            test_clean_set = ds(path, train=True, download=True, transform=transform.test_clean)
            test_clean_set.train = False
            test_clean_set.test_data = test_clean_set.train_data[-5000:]
            test_clean_set.test_labels = test_clean_set.train_labels[-5000:]
            delattr(test_clean_set, 'train_data')
            delattr(test_clean_set, 'train_labels') 
        num_classes = max(train_set.targets) + 1
        align_set = torch.utils.data.Subset(align_set, list(range(10000)))
        test_align_set = test_set
    elif dataset == 'TINYIMAGENET200':
        train_folder = path + '/train'
        if not os.path.exists(train_folder):
            tinyimagenet200_data.download_dataset(path, path_old)
        train_set = datasets.ImageFolder(os.path.join(path, 'train'), transform.train)
        align_set = datasets.ImageFolder(os.path.join(path, 'train'), transform.test)
        align_set = torch.utils.data.Subset(align_set, list(range(10000)))
        tinyimagenet200_data.ensure_dataset_loaded(data_path=path)
        test_set = datasets.ImageFolder(os.path.join(path, 'val_fixed'), transform.test)
        test_clean_set = datasets.ImageFolder(os.path.join(path, 'val_fixed'), transform.test_clean) 
        test_align_set = test_set
        num_classes = 200
    else:
        raise ValueError('Code needs to be generalized for given dataset.')
    return {
               'train': torch.utils.data.DataLoader(
                   train_set,
                   batch_size=batch_size,
                   shuffle=shuffle_train,
                   num_workers=num_workers,
                   pin_memory=True
               ),
               'test': torch.utils.data.DataLoader(
                   test_set,
                   batch_size=test_batch_size,
                   shuffle=False,
                   num_workers=num_workers,
                   pin_memory=True
               ),
               'test_clean': torch.utils.data.DataLoader(
                   test_clean_set,
                   batch_size=test_batch_size,
                   shuffle=False,
                   num_workers=num_workers,
                   pin_memory=True
               ), 
               'align': torch.utils.data.DataLoader(
                    align_set,
                    batch_size=test_batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True
                ),
               'test_align': torch.utils.data.DataLoader(
                   test_align_set,
                   batch_size=batch_size,
                   shuffle=False,
                   num_workers=num_workers,
                   pin_memory=True
               ),
               'testset': test_set,
           }, num_classes
