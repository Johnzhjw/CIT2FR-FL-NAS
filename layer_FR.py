# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

from collections import OrderedDict

import torch.nn as nn
from ofa.utils import MyModule, build_activation, get_same_padding, SEModule, ShuffleLayer
from ofa.layers import MBInvertedConvLayer, ConvLayer, DepthConvLayer, PoolingLayer, IdentityLayer, LinearLayer, ZeroLayer

import torch as th
import math
from torch.nn import init


def set_layer_from_config(layer_config):
    if layer_config is None:
        return None

    name2layer = {
        ConvLayer.__name__: ConvLayer,
        DepthConvLayer.__name__: DepthConvLayer,
        PoolingLayer.__name__: PoolingLayer,
        IdentityLayer.__name__: IdentityLayer,
        FuzzyLayer.__name__: FuzzyLayer,
        LinearLayer.__name__: LinearLayer,
        ZeroLayer.__name__: ZeroLayer,
        MBInvertedConvLayer.__name__: MBInvertedConvLayer,
    }

    layer_name = layer_config.pop('name')
    layer = name2layer[layer_name]
    return layer.build_from_config(layer_config)


class T2FuzzLayer2Dsimp(nn.Module):

    def __init__(self, dim=2):
        super(T2FuzzLayer2Dsimp, self).__init__()
        self.dim = dim
        self.param_c = nn.Parameter(th.rand(1, 1), requires_grad=True)  # Gaussian
        self.param_d = nn.Parameter(th.rand(1, dim), requires_grad=True)  # Gaussian
        init.kaiming_uniform_(self.param_c, a=math.sqrt(5))
        init.kaiming_uniform_(self.param_d, a=math.sqrt(5))

    def forward(self, x):
        x_out =[]
        for i in range(self.dim):
            x_out.append(th.exp(-(x - self.param_c[0, 0]) ** 2 * (2 * self.param_d[0, i] ** 2 + 1e-6)))
        return th.stack(x_out, 4)


class FuzzyLayer(MyModule):

    def __init__(self, in_channels, bias=True,
                 use_bn=False, act_func=None, dropout_rate=0, ops_order='weight_bn_act', flag_layer=False):
        super(FuzzyLayer, self).__init__()

        self.in_channels = in_channels
        self.bias = bias

        self.use_bn = use_bn
        self.act_func = act_func
        self.dropout_rate = dropout_rate
        self.ops_order = ops_order

        self.flag_layer = flag_layer

        """ modules """
        modules = {}
        # batch norm
        if self.use_bn:
            if self.bn_before_weight:
                modules['bn'] = nn.BatchNorm2d(in_channels)
            else:
                modules['bn'] = nn.BatchNorm1d(in_channels)
        else:
            modules['bn'] = None
        # activation
        modules['act'] = build_activation(self.act_func, self.ops_list[0] != 'act')
        # dropout
        if self.dropout_rate > 0:
            modules['dropout'] = nn.Dropout(self.dropout_rate, inplace=True)
        else:
            modules['dropout'] = None
        # linear
        self.dim = 2
        modules['weight'] = {'fuzzy': nn.Sequential(
            T2FuzzLayer2Dsimp(self.dim),
            nn.AdaptiveAvgPool3d([1, 1, self.dim]),
            nn.AvgPool3d([1, 1, self.dim])
        )}

        # add modules
        for op in self.ops_list:
            if modules[op] is None:
                continue
            elif op == 'weight':
                if modules['dropout'] is not None:
                    self.add_module('dropout', modules['dropout'])
                for key in modules['weight']:
                    self.add_module(key, modules['weight'][key])
            else:
                self.add_module(op, modules[op])

    @property
    def get_flag(self):
        return self.flag_layer

    @get_flag.setter
    def get_flag(self, flag):
        self.flag_layer = flag

    @property
    def ops_list(self):
        return self.ops_order.split('_')

    @property
    def bn_before_weight(self):
        for op in self.ops_list:
            if op == 'bn':
                return True
            elif op == 'weight':
                return False
        raise ValueError('Invalid ops_order: %s' % self.ops_order)

    def forward(self, x):
        if self.flag_layer:
            for module in self._modules.values():
                x = module(x)
        return x

    @property
    def module_str(self):
        return '%dx%d_Linear' % (self.in_channels, self.in_channels)

    @property
    def config(self):
        return {
            'name': FuzzyLayer.__name__,
            'in_channels': self.in_channels,
            'bias': self.bias,
            'use_bn': self.use_bn,
            'act_func': self.act_func,
            'dropout_rate': self.dropout_rate,
            'ops_order': self.ops_order,
            'flag_layer': self.flag_layer,
        }

    @staticmethod
    def build_from_config(config):
        return FuzzyLayer(**config)

