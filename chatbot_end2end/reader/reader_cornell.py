#!/usr/bin/env python
import glob, os, yaml, sys, re
from nlptools.utils import zload, zdump, flat_list
from .reader_base import Reader_Base

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

class Reader_Cornell(Reader_Base):
    '''
        Read from babi training data, inherit from Reader_Base

        Input:
            - see Reader_Base
    '''


