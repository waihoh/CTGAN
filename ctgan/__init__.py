# -*- coding: utf-8 -*-

"""Top-level package for ctgan."""

__author__ = 'MIT Data To AI Lab'
__email__ = 'dailabmit@gmail.com'
__version__ = '0.2.3.dev0'

from ctgan.demo import load_demo
from ctgan.ctgan import CTGANSynthesizer
from ctgan.tvae import TVAESynthesizer
from ctgan.tablegan import TableganSynthesizer
from ctgan.tvae_val import TVAESynthesizerVal

__all__ = (
    'CTGANSynthesizer',
    'TVAESynthesizer',
    'TableganSynthesizer',
    'load_demo',
    'TVAESynthesizerVal'
)
