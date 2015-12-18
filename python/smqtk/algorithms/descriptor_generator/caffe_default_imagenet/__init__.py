from .descriptor import CaffeDefaultImageNet
from .descriptor_general import CaffeDescriptorGenerator


__author__ = 'paul.tunison@kitware.com'


DESCRIPTOR_GENERATOR_CLASS = [
    CaffeDefaultImageNet,
    CaffeDescriptorGenerator,
]
