import warnings
import os
import math
import numpy as np

import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from ofa.imagenet_codebase.data_providers.base_provider import DataProvider, MyRandomResizedCrop, MyDistributedSampler


class LC25000DataProvider(DataProvider):

    def __init__(self, save_path=None, train_batch_size=256, test_batch_size=512, valid_size=None, n_worker=32,
                 resize_scale=0.08, distort_color=None, image_size=224, flag_FL=False, size_FL=0,
                 num_replicas=None, rank=None):

        warnings.filterwarnings('ignore')
        self._save_path = save_path

        self.image_size = image_size  # int or list of int
        self.distort_color = distort_color
        self.resize_scale = resize_scale

        self.flag_FL = flag_FL
        self.size_FL = size_FL

        self._valid_transform_dict = {}
        if not isinstance(self.image_size, int):
            assert isinstance(self.image_size, list)
            from ofa.imagenet_codebase.data_providers.my_data_loader import MyDataLoader
            self.image_size.sort()  # e.g., 160 -> 224
            MyRandomResizedCrop.IMAGE_SIZE_LIST = self.image_size.copy()
            MyRandomResizedCrop.ACTIVE_SIZE = max(self.image_size)

            for img_size in self.image_size:
                self._valid_transform_dict[img_size] = self.build_valid_transform(img_size)
            self.active_img_size = max(self.image_size)
            valid_transforms = self._valid_transform_dict[self.active_img_size]
            self.train_loader_class = MyDataLoader  # randomly sample image size for each batch of training image
        else:
            self.active_img_size = self.image_size
            valid_transforms = self.build_valid_transform()
            self.train_loader_class = torch.utils.data.DataLoader

        train_transforms = self.build_train_transform()
        train_dataset = self.train_dataset(train_transforms)

        if valid_size is not None:
            if not isinstance(valid_size, int):
                assert isinstance(valid_size, float) and 0 < valid_size < 1
                valid_size = int(len(train_dataset.samples) * valid_size)

            valid_dataset = self.train_dataset(valid_transforms)
            train_indexes, valid_indexes = self.random_sample_valid_set(len(train_dataset.samples), valid_size)

            if num_replicas is not None:
                train_sampler = MyDistributedSampler(train_dataset, num_replicas, rank, np.array(train_indexes))
                valid_sampler = MyDistributedSampler(valid_dataset, num_replicas, rank, np.array(valid_indexes))
            else:
                train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indexes)
                valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indexes)

            self.train = self.train_loader_class(
                train_dataset, batch_size=train_batch_size, sampler=train_sampler,
                num_workers=n_worker, pin_memory=True,
            )
            self.valid = torch.utils.data.DataLoader(
                valid_dataset, batch_size=test_batch_size, sampler=valid_sampler,
                num_workers=n_worker, pin_memory=True,
            )
        else:
            if num_replicas is not None:
                train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas, rank)
                self.train = self.train_loader_class(
                    train_dataset, batch_size=train_batch_size, sampler=train_sampler,
                    num_workers=n_worker, pin_memory=True
                )
            else:
                self.train = self.train_loader_class(
                    train_dataset, batch_size=train_batch_size, shuffle=True,
                    num_workers=n_worker, pin_memory=True,
                )
            self.valid = None

        test_dataset = self.test_dataset(valid_transforms)
        if num_replicas is not None:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas, rank)
            self.test = torch.utils.data.DataLoader(
                test_dataset, batch_size=test_batch_size, sampler=test_sampler, num_workers=n_worker, pin_memory=True,
            )
        else:
            self.test = torch.utils.data.DataLoader(
                test_dataset, batch_size=test_batch_size, shuffle=True, num_workers=n_worker, pin_memory=True,
            )

        if self.valid is None:
            # self.valid = self.test
            valid_dataset = self.valid_dataset(valid_transforms)
            if num_replicas is not None:
                valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset, num_replicas, rank)
                self.valid = torch.utils.data.DataLoader(
                    valid_dataset, batch_size=test_batch_size, sampler=valid_sampler, num_workers=n_worker, pin_memory=True,
                )
            else:
                self.valid = torch.utils.data.DataLoader(
                    valid_dataset, batch_size=test_batch_size, shuffle=True, num_workers=n_worker, pin_memory=True,
                )

        self.len_train = len(train_dataset.samples)

        if self.flag_FL and self.size_FL > 0:
            indexes = self.uniform_sample_train_set()

            samplers = []
            if num_replicas is not None:
                for _ in range(self.size_FL):
                    samplers.append(MyDistributedSampler(train_dataset, num_replicas, rank, np.array(indexes[_])))
            else:
                for _ in range(self.size_FL):
                    samplers.append(torch.utils.data.sampler.SubsetRandomSampler(indexes[_]))

            self.train_splits = []
            for _ in range(self.size_FL):
                self.train_splits.append(self.train_loader_class(
                    train_dataset, batch_size=train_batch_size, sampler=samplers[_],
                    num_workers=n_worker, pin_memory=True,
                ))

    @staticmethod
    def name():
        return 'LC25000'

    @property
    def n_channels(self):
        return 3

    @property
    def data_shape(self):
        return 3, self.active_img_size, self.active_img_size  # C, H, W

    @property
    def n_classes(self):
        return 5

    @property
    def save_path(self):
        if self._save_path is None:
            # self._save_path = '/dataset/imagenet'
            # self._save_path = '/usr/local/soft/temp-datastore/ILSVRC2012'  # servers
            self._save_path = '/mnt/datastore/LC25000'  # home server

            if not os.path.exists(self._save_path):
                # self._save_path = os.path.expanduser('~/dataset/imagenet')
                # self._save_path = os.path.expanduser('/usr/local/soft/temp-datastore/ILSVRC2012')
                self._save_path = '/mnt/datastore/LC25000'  # home server
        return self._save_path

    @property
    def data_url(self):
        raise ValueError('unable to download %s' % self.name())

    def train_dataset(self, _transforms):
        dataset = datasets.ImageFolder(self.train_path, _transforms)
        return dataset

    def valid_dataset(self, _transforms):
        dataset = datasets.ImageFolder(self.valid_path, _transforms)
        return dataset

    def test_dataset(self, _transforms):
        dataset = datasets.ImageFolder(self.test_path, _transforms)
        return dataset

    @property
    def train_path(self):
        return os.path.join(self.save_path, 'train')

    @property
    def valid_path(self):
        return os.path.join(self.save_path, 'val')

    @property
    def test_path(self):
        return os.path.join(self.save_path, 'test')

    @property
    def normalize(self):
        # return transforms.Normalize(mean=[0.66946244, 0.53382075, 0.851768], std=[0.1291297, 0.17449944, 0.074376434])
        return transforms.Normalize(mean=[0.72835726, 0.59946805, 0.87657416], std=[0.13199076, 0.17347044, 0.06519146])

    def build_train_transform(self, image_size=None, print_log=True):
        if image_size is None:
            image_size = self.image_size
        if print_log:
            print('Color jitter: %s, resize_scale: %s, img_size: %s' %
                  (self.distort_color, self.resize_scale, image_size))

        if self.distort_color == 'torch':
            color_transform = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
        elif self.distort_color == 'tf':
            color_transform = transforms.ColorJitter(brightness=32. / 255., saturation=0.5)
        else:
            color_transform = None

        if isinstance(image_size, list):
            resize_transform_class = MyRandomResizedCrop
            print('Use MyRandomResizedCrop: %s, \t %s' % MyRandomResizedCrop.get_candidate_image_size(),
                  'sync=%s, continuous=%s' % (MyRandomResizedCrop.SYNC_DISTRIBUTED, MyRandomResizedCrop.CONTINUOUS))
        else:
            resize_transform_class = transforms.RandomResizedCrop

        train_transforms = [
            resize_transform_class(image_size, scale=(self.resize_scale, 1.0)),
            transforms.RandomHorizontalFlip(),
        ]
        if color_transform is not None:
            train_transforms.append(color_transform)
        train_transforms += [
            transforms.ToTensor(),
            self.normalize,
        ]

        train_transforms = transforms.Compose(train_transforms)
        return train_transforms

    def build_valid_transform(self, image_size=None):
        if image_size is None:
            image_size = self.active_img_size
        return transforms.Compose([
            transforms.Resize(int(math.ceil(image_size / 0.875))),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            self.normalize,
        ])

    def uniform_sample_train_set(self):
        n_splits = self.size_FL

        g = torch.Generator()
        g.manual_seed(DataProvider.VALID_SEED)  # set random seed before sampling validation set
        rand_indexes = torch.randperm(self.len_train, generator=g).tolist()

        indexes = []
        tmp_offset = 0
        for sz in range(n_splits):
            ind_bg = tmp_offset
            ind_fn = ind_bg + self.len_train // n_splits
            if sz == n_splits-1:
                ind_fn = self.len_train
            indexes.append(rand_indexes[ind_bg:ind_fn])
            tmp_offset = ind_fn
        return indexes

    def assign_active_img_size(self, new_img_size):
        self.active_img_size = new_img_size
        if self.active_img_size not in self._valid_transform_dict:
            self._valid_transform_dict[self.active_img_size] = self.build_valid_transform()
        # change the transform of the valid and test set
        self.valid.dataset.transform = self._valid_transform_dict[self.active_img_size]
        self.test.dataset.transform = self._valid_transform_dict[self.active_img_size]

    def build_sub_train_loader(self, n_images, batch_size, num_worker=None, num_replicas=None, rank=None, tag_FL=-1):
        # used for resetting running statistics
        if self.__dict__.get('sub_train_%d_%d' % (tag_FL+100, self.active_img_size), None) is None:
            if num_worker is None:
                num_worker = self.train.num_workers

            if tag_FL >= 0:
                n_samples = self.train_splits[tag_FL].sampler.__len__()
            else:
                n_samples = self.train.sampler.__len__()
            g = torch.Generator()
            g.manual_seed(DataProvider.SUB_SEED)
            rand_indexes = torch.randperm(n_samples, generator=g).tolist()
            n_images = n_samples if n_samples < n_images else n_images

            new_train_dataset = self.train_dataset(
                self.build_train_transform(image_size=self.active_img_size, print_log=False))

            if tag_FL >= 0:
                indexes = self.uniform_sample_train_set()
                chosen_indexes = [indexes[tag_FL][_] for _ in rand_indexes[:n_images]]
            else:
                chosen_indexes = rand_indexes[:n_images]
            if num_replicas is not None:
                sub_sampler = MyDistributedSampler(new_train_dataset, num_replicas, rank, np.array(chosen_indexes))
            else:
                sub_sampler = torch.utils.data.sampler.SubsetRandomSampler(chosen_indexes)
            sub_data_loader = torch.utils.data.DataLoader(
                new_train_dataset, batch_size=batch_size, sampler=sub_sampler,
                num_workers=num_worker, pin_memory=True,
            )
            self.__dict__['sub_train_%d_%d' % (tag_FL+100, self.active_img_size)] = []
            for images, labels in sub_data_loader:
                self.__dict__['sub_train_%d_%d' % (tag_FL+100, self.active_img_size)].append((images, labels))
        return self.__dict__['sub_train_%d_%d' % (tag_FL+100, self.active_img_size)]
