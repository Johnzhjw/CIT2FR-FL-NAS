import os
import time
import json
import torch
import argparse
import numpy as np

import utils
from codebase.networks import NSGANetV2
from codebase.run_manager import get_run_config
from ofa_mbv3_fr import OFAMobileNetV3
from run_manager import RunManager
from ofa.elastic_nn.modules.dynamic_op import DynamicSeparableConv2d

import warnings
warnings.simplefilter("ignore")

DynamicSeparableConv2d.KERNEL_TRANSFORM_MODE = 1


def parse_string_list(string):
    if isinstance(string, str):
        # convert '[5 5 5 7 7 7 3 3 7 7 7 3 3]' to [5, 5, 5, 7, 7, 7, 3, 3, 7, 7, 7, 3, 3]
        return list(map(int, string[1:-1].split()))
    else:
        return string


def pad_none(x, depth, max_depth):
    new_x, counter = [], 0
    for d in depth:
        for _ in range(d):
            new_x.append(x[counter])
            counter += 1
        if d < max_depth:
            new_x += [None] * (max_depth - d)
    return new_x


def get_net_info(net, data_shape, measure_latency=None, print_info=True, clean=False, lut=None):

    net_info = utils.get_net_info(
        net, data_shape, measure_latency, print_info=print_info, clean=clean, lut=lut)

    gpu_latency, cpu_latency = None, None
    for k in net_info.keys():
        if 'gpu' in k:
            gpu_latency = np.round(net_info[k]['val'], 2)
        if 'cpu' in k:
            cpu_latency = np.round(net_info[k]['val'], 2)

    return {
        'params': np.round(net_info['params'] / 1e6, 2),
        'flops': np.round(net_info['flops'] / 1e6, 2),
        'gpu': gpu_latency, 'cpu': cpu_latency
    }


def validate_config(config, max_depth=3):
    kernel_size, exp_ratio, depth, fuz = config['ks'], config['e'], config['d'], config['f']

    if isinstance(kernel_size, str): kernel_size = parse_string_list(kernel_size)
    if isinstance(exp_ratio, str): exp_ratio = parse_string_list(exp_ratio)
    if isinstance(depth, str): depth = parse_string_list(depth)
    if isinstance(fuz, str): fuz = list(map(int, fuz))[0]

    assert (isinstance(kernel_size, list) or isinstance(kernel_size, int))
    assert (isinstance(exp_ratio, list) or isinstance(exp_ratio, int))
    assert isinstance(depth, list)
    assert isinstance(fuz, int)

    if len(kernel_size) < len(depth) * max_depth:
        kernel_size = pad_none(kernel_size, depth, max_depth)
    if len(exp_ratio) < len(depth) * max_depth:
        exp_ratio = pad_none(exp_ratio, depth, max_depth)

    # return {'ks': kernel_size, 'e': exp_ratio, 'd': depth, 'w': config['w']}
    return {'ks': kernel_size, 'e': exp_ratio, 'd': depth, 'f': fuz}


class OFAEvaluator:
    """ based on OnceForAll supernet taken from https://github.com/mit-han-lab/once-for-all """
    def __init__(self,
                 n_channel_in=3,
                 n_classes=1000,
                 model_path='./data/ofa_mbv3_d234_e346_k357_w1.0',
                 kernel_size=None, exp_ratio=None, depth=None, flag_fuzzy=False,
                 flag_not_image=False, ignore_weights=False, load_w_after_arch_sample=False):
        # default configurations
        if flag_not_image:
            self.kernel_size = [1] if kernel_size is None else kernel_size  # depth-wise conv kernel size
        else:
            self.kernel_size = [3, 5, 7] if kernel_size is None else kernel_size  # depth-wise conv kernel size
        self.exp_ratio = [1, 2, 3, 4] if exp_ratio is None else exp_ratio  # expansion rate
        self.depth = [1, 2, 3, 4] if depth is None else depth  # number of MB block repetition

        self.load_w_after_arch_sample = load_w_after_arch_sample
        self.model_path = model_path

        if 'w1.0' in model_path:
            self.width_mult = 1.0
            if flag_fuzzy:
                self.model_path = model_path.replace('w1.0', 'FR_w1.0')
        elif 'w1.2' in model_path:
            self.width_mult = 1.2
            if flag_fuzzy:
                self.model_path = model_path.replace('w1.2', 'FR_w1.2')
        else:
            raise ValueError

        self.engine = OFAMobileNetV3(
            n_classes=n_classes,
            dropout_rate=0, width_mult_list=self.width_mult, ks_list=self.kernel_size,
            expand_ratio_list=self.exp_ratio, depth_list=self.depth,
            flag_fuzzy=flag_fuzzy, n_channel_in=n_channel_in, flag_not_image=flag_not_image
        )

        if not ignore_weights and not load_w_after_arch_sample:
            init = torch.load(self.model_path, map_location='cpu')['state_dict']
            self.engine.load_weights_from_net(init)

    def sample(self, config=None):
        """ randomly sample a sub-network """
        if config is not None:
            config = validate_config(config, max(self.depth))
            self.engine.set_active_subnet(ks=config['ks'], e=config['e'], d=config['d'], f=config['f'])
        else:
            config = self.engine.sample_active_subnet()

        subnet = self.engine.get_active_subnet(preserve_weight=True)
        return subnet, config

    @staticmethod
    def save_net_config(path, net, config_name='net.config'):
        """ dump run_config and net_config to the model_folder """
        net_save_path = os.path.join(path, config_name)
        json.dump(net.config, open(net_save_path, 'w'), indent=4)
        print('Network configs dump to %s' % net_save_path)

    @staticmethod
    def save_net(path, net, model_name):
        """ dump net weight as checkpoint """
        if isinstance(net, torch.nn.DataParallel):
            checkpoint = {'state_dict': net.module.state_dict()}
        else:
            checkpoint = {'state_dict': net.state_dict()}
        model_path = os.path.join(path, model_name)
        torch.save(checkpoint, model_path)
        print('Network model dump to %s' % model_path)

    @staticmethod
    def eval(run_config, subnet,
             n_epochs=0, resolution=224, is_test=True, log_dir='.tmp/eval', measure_latency=None, no_logs=False,
             reset_running_statistics=True,
             ignore_weights=False, load_w_after_arch_sample=False, model_path=None):

        if not ignore_weights and load_w_after_arch_sample:
            init = torch.load(model_path, map_location='cpu')['state_dict']
            subnet.load_state_dict(init)

        lut = {'cpu': 'data/i7-8700K_lut.yaml'}

        info = get_net_info(
            subnet, (run_config.data_provider.n_channels, resolution, resolution), measure_latency=measure_latency,
            print_info=False, clean=True, lut=lut)

        # set the image size. You can set any image size from 128(192) to 256 here
        run_config.data_provider.assign_active_img_size(resolution)

        if n_epochs > 0:
            # for datasets other than the one supernet was trained on (ImageNet)
            # a few epochs of training need to be applied
            # subnet.reset_classifier(
            #     last_channel=subnet.classifier.in_features,
            #     n_classes=run_config.data_provider.n_classes, dropout_rate=cfgs.drop_rate)
            NSGANetV2.reset_classifier(
                subnet,
                last_channel=subnet.classifier.in_features,
                n_classes=run_config.data_provider.n_classes, dropout_rate=cfgs.drop_rate)

        run_manager = RunManager(log_dir, subnet, run_config, init=False)
        subnet = run_manager.network
        # return
        '''
        print(subnet==run_manager.network)
        print(subnet.classifier==run_manager.network.classifier)
        print(subnet.classifier.linear==run_manager.network.classifier.linear)
        print(subnet.classifier.linear.weight==run_manager.network.classifier.linear.weight)
        '''
        if reset_running_statistics:
            # run_manager.reset_running_statistics(net=subnet, batch_size=vld_batch_size)
            run_manager.reset_running_statistics(net=subnet)
            if n_epochs == 0 and run_manager.run_config.flag_FL:
                run_manager.init_FL(reset_running_statistics)
                run_manager.FedAvg()
                subnet = run_manager.network


        if n_epochs > 0:
            subnet = run_manager.train(cfgs, flag_reset_running_statistics=reset_running_statistics)

        loss, top1, top5 = [], [], []
        t_loss, t_top1 = run_manager.validate(net=subnet, is_test=is_test, no_logs=False)
        loss.append(t_loss)
        top1.append(t_top1)
        if not is_test:
            t_loss, t_top1 = run_manager.validate(net=subnet, is_test=True, no_logs=False)
            loss.append(t_loss)
            top1.append(t_top1)

        info['loss'], info['top1'] = loss, top1

        save_path = os.path.join(log_dir, 'net.stats') if cfgs.save is None else cfgs.save
        if cfgs.save_config:
            dirname, filename = os.path.split(cfgs.subnet)
            log_path = os.path.join(log_dir, dirname)
            os.makedirs(log_path, exist_ok=True)
            OFAEvaluator.save_net_config(log_path, subnet, filename+"net.config")
            OFAEvaluator.save_net(log_path, subnet, filename+"net.init")
        with open(save_path, 'w') as handle:
            json.dump(info, handle)

        print(info)


def main(args):

    run_config = get_run_config(
        dataset=args.dataset, data_path=args.data, image_size=args.resolution, n_epochs=args.n_epochs,
        train_batch_size=args.trn_batch_size, test_batch_size=args.vld_batch_size,
        n_worker=args.num_workers, valid_size=args.valid_size, flag_FL=args.flag_FL, size_FL=args.size_FL)

    n_channel_in = 3
    n_classes = 1000
    if 'CerebralInfarction.csv' in args.data:
        n_channel_in = 60
        n_classes = 2
    elif 'ALF_Data.csv' in args.data:
        n_channel_in = 28
        n_classes = 2
    elif 'LC25000' in args.data:
        n_channel_in = 3
        n_classes = 5

    """ one evaluation of a subnet or a config from a file """
    mode = 'subnet'
    if args.config is not None:
        if args.init is not None and not args.ignore_weights:
            mode = 'config'
        if args.ignore_weights:
            mode = 'config'

    flag_fuzzy = args.flag_fuzzy

    if n_channel_in != run_config.data_provider.n_channels:
        print('The number of input channels is wrong (%d != %d), please check!' %
              (n_channel_in, run_config.data_provider.n_channels))
    if n_classes != run_config.data_provider.n_classes:
        print('The number of output classes is wrong (%d != %d), please check!' %
              (n_classes, run_config.data_provider.n_classes))

    model_path = args.supernet_path

    print('Evaluation mode: {}'.format(mode))
    if mode == 'config':
        net_config = json.load(open(args.config))
        subnet = NSGANetV2.build_from_config(net_config, drop_connect_rate=args.drop_connect_rate)
        if not args.ignore_weights:
            init = torch.load(args.init, map_location='cpu')['state_dict']
            subnet.load_state_dict(init)
        subnet.classifier.dropout_rate = args.drop_rate
        # flag_fuzzy = subnet.fuzzy_layer.get_flag
        try:
            resolution = net_config['resolution']
        except KeyError:
            resolution = args.resolution

    elif mode == 'subnet':
        config = json.load(open(args.subnet))
        flag_fuzzy = config['f']
        evaluator = OFAEvaluator(n_channel_in=run_config.data_provider.n_channels,
                                 n_classes=run_config.data_provider.n_classes,
                                 model_path=args.supernet_path, flag_fuzzy=flag_fuzzy,
                                 flag_not_image=args.flag_not_image, ignore_weights=args.ignore_weights,
                                 load_w_after_arch_sample=args.load_w_after_arch_sample)
        subnet, _ = evaluator.sample({'ks': config['ks'], 'e': config['e'], 'd': config['d'], 'f': config['f']})
        resolution = config['r']
        model_path = evaluator.model_path

    else:
        raise NotImplementedError

    if args.flag_not_image:
        resolution = 1

    OFAEvaluator.eval(
        run_config, subnet, log_dir=args.log_dir,
        n_epochs=args.n_epochs,
        resolution=resolution, is_test=args.test, measure_latency=args.latency,
        no_logs=(not args.verbose), reset_running_statistics=args.reset_running_statistics,
        ignore_weights=args.ignore_weights, load_w_after_arch_sample=args.load_w_after_arch_sample,
        model_path=model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='/mnt/datastore/ILSVRC2012',
                        help='location of the data corpus')
    parser.add_argument('--log_dir', type=str, default='.tmp',
                        help='directory for logging')
    parser.add_argument('--str_time', type=str, default=None,
                        help='time string')
    parser.add_argument('--dataset', type=str, default='imagenet',
                        help='name of the dataset (imagenet, cifar10, cifar100, ...)')
    parser.add_argument('--n_classes', type=int, default=1000,
                        help='number of classes for the given dataset')
    parser.add_argument('--supernet_path', type=str, default='./data/ofa_mbv3_d234_e346_k357_w1.0',
                        help='file path to supernet weights')
    parser.add_argument('--ignore_weights', action='store_true', default=False,
                        help='whether or not use federated learning')
    parser.add_argument('--subnet', type=str, default=None,
                        help='location of a json file of ks, e, d, and e')
    parser.add_argument('--config', type=str, default=None,
                        help='location of a json file of specific model declaration')
    parser.add_argument('--init', type=str, default=None,
                        help='location of initial weight to load')
    parser.add_argument('--trn_batch_size', type=int, default=128,
                        help='test batch size for inference')
    parser.add_argument('--vld_batch_size', type=int, default=256,
                        help='test batch size for inference')
    parser.add_argument('--num_workers', type=int, default=6,
                        help='number of workers for data loading')
    parser.add_argument('--n_epochs', type=int, default=0,
                        help='number of training epochs')
    parser.add_argument('--save', type=str, default=None,
                        help='location to save the evaluated metrics')
    parser.add_argument('--resolution', type=int, default=224,
                        help='input resolution (192 -> 256)')
    parser.add_argument('--valid_size', type=int, default=None,
                        help='validation set size, randomly sampled from training set')
    parser.add_argument('--test', action='store_true', default=False,
                        help='evaluation performance on testing set')
    parser.add_argument('--latency', type=str, default=None,
                        help='latency measurement settings (gpu64#cpu)')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='whether to display evaluation progress')
    parser.add_argument('--reset_running_statistics', action='store_true', default=False,
                        help='reset the running mean / std of BN')
    parser.add_argument('--drop_rate', type=float, default=0.2,
                        help='dropout rate')
    parser.add_argument('--drop_connect_rate', type=float, default=0.0,
                        help='connection dropout rate')
    parser.add_argument('--save_config', action='store_true', default=True,
                        help='save config file')
    parser.add_argument('--flag_fuzzy', action='store_true', default=False,
                        help='whether or not use fuzzy')
    parser.add_argument('--flag_FL', action='store_true', default=False,
                        help='whether or not use federated learning')
    parser.add_argument('--size_FL', type=int, default=0,
                        help='number of clients for federated learning')
    parser.add_argument('--flag_not_image', action='store_true', default=False,
                        help='The inputs are not images')
    parser.add_argument('--load_w_after_arch_sample', action='store_true', default=False,
                        help='Load weights after the architecture is sampled')
    cfgs = parser.parse_args()

    if cfgs.str_time is None:
        cfgs.str_time = time.strftime("%Y%m%d-%H%M%S")
    cfgs.log_dir = '{}-{}'.format(cfgs.log_dir, cfgs.str_time)

    cfgs.teacher_model = None

    print(cfgs)

    main(cfgs)

