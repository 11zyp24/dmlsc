import torch
import random
import numpy as np
import os, sys
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Sampler
# from .base_data_loader import BaseDataLoader
from PIL import Image
from .imbalance_cifar import IMBALANCECIFAR10, IMBALANCECIFAR100, DirichletImbalanceCIFAR10, DirichletImbalanceCIFAR100


class CIFAR100DataLoader(DataLoader):
    """
    Load CIFAR 100  该类用于加载CIFAR-100数据集。根据training标志，决定是加载训练集还是测试集
    """
    def __init__(self, data_dir, batch_size, shuffle=True, num_workers=1, training=True):
        normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
            std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
        train_trsfm = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize,
        ])
        test_trsfm = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        if training:
            self.dataset = datasets.CIFAR100(data_dir, train=training, download=True, transform=train_trsfm)
            self.val_dataset = datasets.CIFAR100(data_dir, train=False, download=True, transform=test_trsfm) # test set
        else:
            self.dataset = datasets.CIFAR100(data_dir, train=training, download=True, transform=test_trsfm)
        
        num_classes = len(np.unique(self.dataset.targets))
        assert num_classes == 100

        cls_num_list = [0] * num_classes
        for label in self.dataset.targets:
            cls_num_list[label] += 1

        self.cls_num_list = cls_num_list

        self.shuffle = shuffle
        self.n_samples = len(self.dataset)

        self.init_kwargs = {
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'num_workers': num_workers
        }
        super().__init__(dataset=self.dataset, **self.init_kwargs)
        
    def split_validation(self): #分割验证集
        # If you do not want to validate:
        # return None
        # If you want to validate:
        return DataLoader(dataset=self.val_dataset, **self.init_kwargs)

#该类用于平衡采样，将样本按照类别分组（buckets），然后在每次训练时，随机从这些类别组中选择样本，以确保每个类别的样本在 训练过程中都有机会出现
class BalancedSampler(Sampler):
    def __init__(self, buckets, retain_epoch_size=False):
        for bucket in buckets:
            random.shuffle(bucket)

        self.bucket_num = len(buckets)
        self.buckets = buckets
        self.bucket_pointers = [0 for _ in range(self.bucket_num)]
        self.retain_epoch_size = retain_epoch_size
    
    def __iter__(self):
        count = self.__len__()
        while count > 0:
            yield self._next_item()
            count -= 1

    def _next_item(self):
        bucket_idx = random.randint(0, self.bucket_num - 1)
        bucket = self.buckets[bucket_idx]
        item = bucket[self.bucket_pointers[bucket_idx]]
        self.bucket_pointers[bucket_idx] += 1
        if self.bucket_pointers[bucket_idx] == len(bucket):
            self.bucket_pointers[bucket_idx] = 0
            random.shuffle(bucket)
        return item

    def __len__(self):
        if self.retain_epoch_size:
            return sum([len(bucket) for bucket in self.buckets]) # AcruQRally we need to upscale to next full batch
        else:
            return max([len(bucket) for bucket in self.buckets]) * self.bucket_num # Ensures every instance has the chance to be visited in an epoch

class ImbalanceCIFAR100DataLoader(DataLoader):
    """
    Imbalance Cifar100 Data Loader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, num_workers=1, training=True, balanced=False, retain_epoch_size=True, imb_type='exp', imb_factor=0.01):
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010])
        train_trsfm = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize,
        ])
        test_trsfm = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        test_dataset = datasets.CIFAR100(data_dir, train=False, download=True, transform=test_trsfm) # test set
        
        if training:
            dataset = IMBALANCECIFAR100(data_dir, train=True, download=True, transform=train_trsfm, imb_type=imb_type, imb_factor=imb_factor)
            val_dataset = test_dataset
        else:
            dataset = test_dataset
            val_dataset = None

        self.dataset = dataset
        self.val_dataset = val_dataset

        num_classes = len(np.unique(dataset.targets))
        assert num_classes == 100

        cls_num_list = [0] * num_classes
        for label in dataset.targets:
            cls_num_list[label] += 1

        self.cls_num_list = cls_num_list

        if balanced:
            if training:
                buckets = [[] for _ in range(num_classes)]
                for idx, label in enumerate(dataset.targets):
                    buckets[label].append(idx)
                sampler = BalancedSampler(buckets, retain_epoch_size)
                shuffle = False
            else:
                print("Test set will not be evaluated with balanced sampler, nothing is done to make it balanced")
        else:
            sampler = None
        
        self.shuffle = shuffle
        self.init_kwargs = {
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'num_workers': num_workers
        }

        super().__init__(dataset=self.dataset, **self.init_kwargs, sampler=sampler) # Note that sampler does not apply to validation set

    def split_validation(self):
        # If you do not want to validate:
        # return None
        # If you want to validate:
        return DataLoader(dataset=self.val_dataset, **self.init_kwargs)

class  TestAgnosticImbalanceCIFAR100DataLoader(DataLoader):
    """
    Imbalance Cifar100 Data Loader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, num_workers=1, training=True, balanced=False, retain_epoch_size=True, imb_type='exp', imb_factor=0.01, test_imb_factor=0, reverse=False):
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010])
        train_trsfm = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize,
        ])
        test_trsfm = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        test_dataset = datasets.CIFAR100(data_dir, train=False, download=True, transform=test_trsfm) # test set
        
        if training:
            dataset = IMBALANCECIFAR100(data_dir, train=True, download=True, transform=train_trsfm, imb_type=imb_type, imb_factor=imb_factor)
            val_dataset = test_dataset
        else:
            if test_imb_factor!=0:
                dataset = IMBALANCECIFAR100(data_dir, train=False, download=True, transform=train_trsfm, imb_type=imb_type, imb_factor=test_imb_factor, reverse=reverse)
            else:
                dataset = test_dataset
            val_dataset = None

        self.dataset = dataset
        self.val_dataset = val_dataset

        num_classes = len(np.unique(dataset.targets))
        assert num_classes == 100

        cls_num_list = [0] * num_classes
        for label in dataset.targets:
            cls_num_list[label] += 1

        self.cls_num_list = cls_num_list

        if balanced:
            if training:
                buckets = [[] for _ in range(num_classes)]
                for idx, label in enumerate(dataset.targets):
                    buckets[label].append(idx)
                sampler = BalancedSampler(buckets, retain_epoch_size)
                shuffle = False
            else:
                print("Test set will not be evaluated with balanced sampler, nothing is done to make it balanced")
        else:
            sampler = None
        
        self.shuffle = shuffle
        self.init_kwargs = {
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'num_workers': num_workers
        }

        super().__init__(dataset=self.dataset, **self.init_kwargs, sampler=sampler) # Note that sampler does not apply to validation set

    def split_validation(self):
        # If you do not want to validate:
        # return None
        # If you want to validate:
        return DataLoader(dataset=self.val_dataset, **self.init_kwargs)

class ImbalanceCIFAR10DataLoader(DataLoader):
    """
    Imbalance Cifar10 Data Loader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, num_workers=1, training=True, balanced=False, retain_epoch_size=True, imb_factor=0.01):
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010])
        train_trsfm = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize,
        ])
        test_trsfm = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        
        if training:
            dataset = IMBALANCECIFAR10(data_dir, train=True, download=True, transform=train_trsfm, imb_factor=0.01)
            val_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_trsfm) # test set
        else:
            dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_trsfm) # test set
            val_dataset = None

        self.dataset = dataset
        self.val_dataset = val_dataset

        num_classes = len(np.unique(dataset.targets))
        assert num_classes == 10

        cls_num_list = [0] * num_classes
        for label in dataset.targets:
            cls_num_list[label] += 1

        self.cls_num_list = cls_num_list

        if balanced:
            if training:
                buckets = [[] for _ in range(num_classes)]
                for idx, label in enumerate(dataset.targets):
                    buckets[label].append(idx)
                sampler = BalancedSampler(buckets, retain_epoch_size)
                shuffle = False
            else:
                print("Test set will not be evaluated with balanced sampler, nothing is done to make it balanced")
        else:
            sampler = None
        
        self.shuffle = shuffle
        self.init_kwargs = {
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'num_workers': num_workers
        }

        super().__init__(dataset=self.dataset, **self.init_kwargs, sampler=sampler) # Note that sampler does not apply to validation set

    def split_validation(self):
        # If you do not want to validate:
        # return None
        # If you want to validate:
        return DataLoader(dataset=self.val_dataset, **self.init_kwargs)

class DirichletImbalanceCIFAR100DataLoader(DataLoader):  # 用于加载基于 Dirichlet 分布的不平衡 CIFAR-100 数据集
    """
    Imbalance Cifar100 Data Loader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, num_workers=1, training=True, balanced=False,
                 retain_epoch_size=True, imb_type='exp', imb_factor=0.01, test_imb_factor=0, reverse=False):
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2023, 0.1994, 0.2010])
        train_trsfm = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize,
        ])
        test_trsfm = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        test_dataset = datasets.CIFAR100(data_dir, train=False, download=True, transform=test_trsfm)  # test set

        if training:
            dataset = DirichletImbalanceCIFAR100(data_dir, train=True, download=True, transform=train_trsfm,
                                                 imb_type=imb_type, imb_factor=imb_factor)
            val_dataset = test_dataset
        else:
            if test_imb_factor != 0:
                dataset = DirichletImbalanceCIFAR100(data_dir, train=False, download=True, transform=train_trsfm,
                                                     imb_type=imb_type, imb_factor=test_imb_factor, reverse=reverse)
            else:
                dataset = test_dataset
            val_dataset = None

        self.dataset = dataset
        self.val_dataset = val_dataset

        num_classes = len(np.unique(dataset.targets))
        assert num_classes == 100

        cls_num_list = [0] * num_classes
        for label in dataset.targets:
            cls_num_list[label] += 1

        self.cls_num_list = cls_num_list

        if balanced:
            if training:
                buckets = [[] for _ in range(num_classes)]
                for idx, label in enumerate(dataset.targets):
                    buckets[label].append(idx)
                sampler = BalancedSampler(buckets, retain_epoch_size)
                shuffle = False
            else:
                print("Test set will not be evaluated with balanced sampler, nothing is done to make it balanced")
        else:
            sampler = None

        self.shuffle = shuffle
        self.init_kwargs = {
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'num_workers': num_workers
        }

        super().__init__(dataset=self.dataset, **self.init_kwargs,
                         sampler=sampler)  # Note that sampler does not apply to validation set

    def split_validation(self):
        # If you do not want to validate:
        # return None
        # If you want to validate:
        return DataLoader(dataset=self.val_dataset, **self.init_kwargs)


from .cui_cifar import CIFAR100_BASE
from .autoaug import CIFAR10Policy, Cutout
#该函数将标签y转换为One-Hot编码。可以指定类别数量num_classes，如果没有则通过y中最大类别数确定
def to_categorical(y, num_classes=None):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

class CIFAR100DataLoaderNCL(DataLoader):
    """
    Imbalance Cifar100 Data Loader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, num_workers=1, training=True, balanced=False,
                 retain_epoch_size=True, imb_type='exp', imb_factor=0.01):
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2023, 0.1994, 0.2010])
        train_trsfm = transforms.Compose([
                # transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                CIFAR10Policy(),  # add AutoAug
                transforms.ToTensor(),
                Cutout(n_holes=1, length=16),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        test_trsfm = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        test_dataset = CIFAR100_BASE(mode='valid', transform=test_trsfm, imb=imb_factor)  # test set

        if training:
            dataset = CIFAR100_BASE(mode='train', transform=train_trsfm, imb=imb_factor)
            val_dataset = test_dataset
        else:
            dataset = test_dataset
            val_dataset = None

        self.dataset = dataset
        self.val_dataset = val_dataset
        dataset.targets, self.cls_num_list = dataset.collect_labels()
        self.dataset.targets = to_categorical(dataset.targets, num_classes=100)
        num_classes = 100
        if balanced:
            if training:
                buckets = [[] for _ in range(num_classes)]
                for idx, label in enumerate(dataset.targets):
                    buckets[label].append(idx)
                sampler = BalancedSampler(buckets, retain_epoch_size)
                shuffle = False
            else:
                print("Test set will not be evaluated with balanced sampler, nothing is done to make it balanced")
                sampler = None
        else:
            sampler = None

        self.shuffle = shuffle
        self.init_kwargs = {
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'num_workers': num_workers
        }

        super().__init__(dataset=self.dataset, **self.init_kwargs,
                         sampler=sampler)  # Note that sampler does not apply to validation set

        # self._debug()

    def _debug(self):
        a = self.dataset[0]

    def split_validation(self):
        # If you do not want to validate:
        # return None
        # If you want to validate:
        return DataLoader(dataset=self.val_dataset, **self.init_kwargs)