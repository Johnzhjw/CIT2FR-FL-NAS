import warnings
import os
import math
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset, sampler, distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from ofa.imagenet_codebase.data_providers.base_provider import DataProvider, MyRandomResizedCrop, MyDistributedSampler

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder


# 重写Dataset
class Mydataset(Dataset):

    def __init__(self, xx, yy, transform=None):
        self.x = xx
        self.y = yy
        self.tranform = transform

    def __getitem__(self, index):
        x1 = self.x[index]
        y1 = self.y[index]
        if self.tranform != None:
            return self.tranform(x1).reshape(-1, 1, 1), y1[0]
        return x1.reshape(-1, 1, 1), y1[0]

    def __len__(self):
        return len(self.x)


def get_dataset(file_name='../dataset/Stk.0941.HK.all.csv', train_ratio=0.6, valid_ratio=0.2, col_name_tar="Close",
                tag_norm='min-max', train_transforms=None, valid_transforms=None):
    df = pd.read_csv(file_name)
    # df = df.sort_index(ascending=True)
    print(df.head(5))

    df = df[df[col_name_tar].notna()]

    train_len = int(len(df) * train_ratio)
    valid_len = int(len(df) * valid_ratio)

    df = df[[col for i, col in enumerate(df.columns) if i]]
    # 处理缺失值
    imp_median = SimpleImputer(strategy="median")  # 用中位数填补
    imp_freq = SimpleImputer(strategy="most_frequent")  # 用众数填补
    for i, col in enumerate(df.columns):
        if isinstance(df[col][0], str):
            df[col] = imp_freq.fit_transform(df[col].values.reshape(-1, 1))
        else:
            df[col] = imp_median.fit_transform(df[col].values.reshape(-1, 1))
    '''
    # 将 label 列 转为 数值
    le = LabelEncoder()
    df[col_name_tar] = le.fit_transform(df[col_name_tar])
    # 字符串 用 OneHot 编码
    df_new = df
    for i, col in enumerate(df.columns):
        if isinstance(df[col][0], str):
            x = df[col].values.reshape(-1, 1)
            enc = OneHotEncoder(categories='auto').fit(x)
            result = enc.transform(x).toarray()
            ncol_n = result.shape[1]
            cols_o = df_new.columns
            df_new = pd.concat([df_new, pd.DataFrame(result)], axis=1)
            df_new.columns = [c for c in cols_o] + ['%s_%d' % (col, _) for _ in range(ncol_n)]
            df_new.drop(col, axis=1, inplace=True)
    df = df_new
    '''
    le = LabelEncoder()
    for i, col in enumerate(df.columns):
        if isinstance(df[col][0], str):
            df[col] = le.fit_transform(df[col])
    # 提取open,close,high,low,vol 作为feature,并做标准化
    # df = df[["Open", "Close", "High", "Low", "Volume", "Adjusted"]]
    if tag_norm == 'min-max':
        df = df.apply(lambda x: (x - min(x[:train_len])) / (max(x[:train_len]) - min(x[:train_len])))
    elif tag_norm == 'Z-score':
        df = df.apply(lambda x: (x - x[:train_len].mean()) / x[:train_len].std())
    elif tag_norm == 'none':
        df = df.apply(lambda x: x)
    else:
        print('Invalid norm type.')
    df = df.fillna(0)
    df.replace([np.inf, -np.inf], 0, inplace=True)

    in_inds = [i for i, col in enumerate(df.columns) if col != col_name_tar]
    out_inds = [i for i, col in enumerate(df.columns) if col == col_name_tar]
    X = []
    Y = []
    for i in range(df.shape[0]):
        X.append(np.array(df.iloc[i, in_inds], dtype=np.float32).reshape(1, -1))
        Y.append(np.array(df.iloc[i, out_inds], dtype=np.float32).reshape(-1, ))

    print(X[0])
    print(Y[0])

    # # 构建batch
    trainx, trainy = X[:train_len], Y[:train_len]
    validx, validy = X[train_len:train_len + valid_len], Y[train_len:train_len + valid_len]
    testx, testy = X[train_len + valid_len:], Y[train_len + valid_len:]
    train_dataset = Mydataset(trainx, trainy, transform=train_transforms)
    valid_dataset = Mydataset(validx, validy, transform=valid_transforms)
    test_dataset = Mydataset(testx, testy, transform=valid_transforms)
    '''
    train_loader = DataLoader(dataset=Mydataset(trainx, trainy, transform=transforms.ToTensor()),
                              batch_size=trn_batch_size, shuffle=shuffle)
    valid_loader = DataLoader(dataset=Mydataset(validx, validy), batch_size=vld_batch_size, shuffle=shuffle)
    test_loader = DataLoader(dataset=Mydataset(testx, testy), batch_size=vld_batch_size, shuffle=shuffle)
    '''

    return train_dataset, valid_dataset, test_dataset


class TableDataProvider(DataProvider):

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
            self.train_loader_class = DataLoader

        train_transforms = self.build_train_transform()

        train_dataset, valid_dataset, test_dataset = \
            get_dataset(file_name=self.save_path, col_name_tar=self.col_name_tar,
                        train_transforms=train_transforms, valid_transforms=valid_transforms)

        if valid_size is not None:
            if not isinstance(valid_size, int):
                assert isinstance(valid_size, float) and 0 < valid_size < 1
                valid_size = int(len(train_dataset.x) * valid_size)

            # valid_dataset = self.train_dataset(valid_transforms)
            valid_dataset = train_dataset
            train_indexes, valid_indexes = self.random_sample_valid_set(len(train_dataset.x), valid_size)

            if num_replicas is not None:
                train_sampler = MyDistributedSampler(train_dataset, num_replicas, rank, np.array(train_indexes))
                valid_sampler = MyDistributedSampler(valid_dataset, num_replicas, rank, np.array(valid_indexes))
            else:
                train_sampler = sampler.SubsetRandomSampler(train_indexes)
                valid_sampler = sampler.SubsetRandomSampler(valid_indexes)

            self.train = self.train_loader_class(
                train_dataset, batch_size=train_batch_size, sampler=train_sampler,
                num_workers=n_worker, pin_memory=True,
            )
            self.valid = DataLoader(
                valid_dataset, batch_size=test_batch_size, sampler=valid_sampler,
                num_workers=n_worker, pin_memory=True,
            )
        else:
            if num_replicas is not None:
                train_sampler = distributed.DistributedSampler(train_dataset, num_replicas, rank)
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

        # test_dataset = self.test_dataset(valid_transforms)
        if num_replicas is not None:
            test_sampler = distributed.DistributedSampler(test_dataset, num_replicas, rank)
            self.test = DataLoader(
                test_dataset, batch_size=test_batch_size, sampler=test_sampler, num_workers=n_worker, pin_memory=True,
            )
        else:
            self.test = DataLoader(
                test_dataset, batch_size=test_batch_size, shuffle=True, num_workers=n_worker, pin_memory=True,
            )

        if self.valid is None:
            # self.valid = self.test
            # valid_dataset = self.valid_dataset(valid_transforms)
            if num_replicas is not None:
                valid_sampler = distributed.DistributedSampler(valid_dataset, num_replicas, rank)
                self.valid = DataLoader(
                    valid_dataset, batch_size=test_batch_size, sampler=valid_sampler, num_workers=n_worker,
                    pin_memory=True,
                )
            else:
                self.valid = DataLoader(
                    valid_dataset, batch_size=test_batch_size, shuffle=True, num_workers=n_worker, pin_memory=True,
                )

        self.len_train = len(train_dataset.x)
        self.n_ft = train_dataset.x[0].shape[1]

        if self.flag_FL and self.size_FL > 0:
            indexes = self.uniform_sample_train_set()

            samplers = []
            if num_replicas is not None:
                for _ in range(self.size_FL):
                    samplers.append(MyDistributedSampler(train_dataset, num_replicas, rank, np.array(indexes[_])))
            else:
                for _ in range(self.size_FL):
                    samplers.append(sampler.SubsetRandomSampler(indexes[_]))

            self.train_splits = []
            for _ in range(self.size_FL):
                self.train_splits.append(self.train_loader_class(
                    train_dataset, batch_size=train_batch_size, sampler=samplers[_],
                    num_workers=n_worker, pin_memory=True,
                ))

    @staticmethod
    def name():
        return 'Table_csv'

    @property
    def n_channels(self):
        return self.n_ft

    @property
    def data_shape(self):
        return self.n_ft, 1, 1  # C, H, W

    @property
    def n_classes(self):
        return 2

    @property
    def save_path(self):
        if self._save_path is None:
            self._save_path = '/mnt/datastore/Table_csv'  # home server

            if not os.path.exists(self._save_path):
                self._save_path = '/mnt/datastore/Table_csv'  # home server
        return self._save_path

    @property
    def col_name_tar(self):
        if 'ALF' in self.save_path:
            tar = 'HyperTension'  #
        elif 'CerebralInfarction' in self.save_path:
            tar = 'CerebralInfarction'
        else:
            print('Unknown data path.')
        return tar

    @property
    def data_url(self):
        raise ValueError('unable to download %s' % self.name())

    def train_dataset(self, _transforms1, _transforms2):
        dataset, a, b = \
            get_dataset(file_name=self.save_path, col_name_tar=self.col_name_tar,
                        train_transforms=_transforms1, valid_transforms=_transforms2)
        return dataset

    def valid_dataset(self, _transforms1, _transforms2):
        a, dataset, b = \
            get_dataset(file_name=self.save_path, col_name_tar=self.col_name_tar,
                        train_transforms=_transforms1, valid_transforms=_transforms2)
        return dataset

    def test_dataset(self, _transforms1, _transforms2):
        a, b, dataset = \
            get_dataset(file_name=self.save_path, col_name_tar=self.col_name_tar,
                        train_transforms=_transforms1, valid_transforms=_transforms2)
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
        return transforms.Normalize(mean=[0.66946244, 0.53382075, 0.851768], std=[0.1291297, 0.17449944, 0.074376434])

    def build_train_transform(self, image_size=None, print_log=True):
        if image_size is None:
            image_size = self.active_img_size
        train_transforms = [
            transforms.ToTensor(),
        ]
        train_transforms = transforms.Compose(train_transforms)
        return train_transforms

    def build_valid_transform(self, image_size=None):
        if image_size is None:
            image_size = self.active_img_size
        return transforms.Compose([
            transforms.ToTensor(),
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
            if sz == n_splits - 1:
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
        if self.__dict__.get('sub_train_%d_%d' % (tag_FL + 100, self.active_img_size), None) is None:
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
                self.build_train_transform(image_size=self.active_img_size, print_log=False),
                self.build_valid_transform(image_size=self.active_img_size)
            )

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
            self.__dict__['sub_train_%d_%d' % (tag_FL + 100, self.active_img_size)] = []
            for images, labels in sub_data_loader:
                self.__dict__['sub_train_%d_%d' % (tag_FL + 100, self.active_img_size)].append((images, labels))
        return self.__dict__['sub_train_%d_%d' % (tag_FL + 100, self.active_img_size)]
