import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
import argparse
import time


def getStat(data, num_workers=8):
    '''
    Compute mean and variance for training data
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    '''
    device = 'cuda'
    size_img = 128

    ##
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    bg = time.clock()

    print('Compute mean and variance for training data.')
    train_data = ImageFolder(root=data, transform=transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.Resize(size_img),
        transforms.ToTensor()
    ]))
    print(len(train_data))

    end.record()
    fn = time.clock()
    torch.cuda.synchronize()
    print('time time:  ', fn - bg)
    print('torch time: ', start.elapsed_time(end))

    start.record()
    bg = time.clock()

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=128, shuffle=False, num_workers=num_workers,
        pin_memory=True)

    end.record()
    fn = time.clock()
    torch.cuda.synchronize()
    print('time time:  ', fn - bg)
    print('torch time: ', start.elapsed_time(end))

    mean = torch.zeros(3)
    std = torch.zeros(3)
    for step, (inputs, targets) in enumerate(train_loader):
        X, targets = inputs.to(device), targets.to(device)

        for i in range(X.shape[0]):
            for d in range(3):
                mean[d] += X[i, d, :, :].mean()
                std[d] += X[i, d, :, :].std()

    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='../data/LC25000/train/',
                        help='location of the data corpus')
    parser.add_argument('--n_workers', type=int, default=4,
                        help='number of workers for dataloader per evaluation job')
    cfgs = parser.parse_args()

    print(getStat(cfgs.data, cfgs.n_workers))
