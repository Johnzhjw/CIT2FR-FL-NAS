import os
import sys
dirname, filename = os.path.split(os.path.abspath(sys.argv[0]))
if '\\' in dirname:
    sys.path.insert(0, '\\'.join(dirname.split('\\')[:-2]))
if '/' in dirname:
    sys.path.insert(0, '/'.join(dirname.split('/')[:-2]))

from ofa.imagenet_codebase.networks.proxyless_nets import ProxylessNASNets, proxyless_base, MobileNetV2
from mbv3_fr import MobileNetV3, MobileNetV3Large
from codebase.networks.nsganetv2 import NSGANetV2

