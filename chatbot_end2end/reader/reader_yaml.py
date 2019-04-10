#!/usr/bin/env python
import glob, os, yaml, sys, re
from nlptools.utils import setLogger
from .reader_base import ReaderBase

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

class ReaderYAML(ReaderBase):
    '''
        Read from yaml dialogs, inherit from Reader_Base

        Input:
            - see Reader_Base
    '''
    def __init__(self, **args):
        ReaderBase.__init__(self, **args)

    def read(self, yamlfile, bidirectional=False):
        '''
            Input:
                - yamlfile: the yaml file path
                - bidirectional: bool, check if train bidirectional dialog, default is False
        '''
        cached_data = filepath + '.h5'
        if os.path.exists(cached_data) and os.path.getsize(cached_data) > 10240:
            self.data = h5py.File(cached_data, 'r')
            return

        def convs_iter():
            with open(yamlfile, 'r') as f:
                data = yaml.load(f)
            key = ''.join(data['categories'])
            data_raw = data['conversations']
            for c in data_raw: 
                conv = []
                for i in range(0,len(c),2):
                    if i+1 < len(c):
                        conv.append([c[i], c[i+1]])
                    else:
                        conv.append([c[i], '<SILENCE>'])
                yield conv
                if bidirectional:
                    conv = []
                    for i in range(0,len(c),2):
                        if i-1 < 0:
                            conv.append(['<SILENCE>', c[i]])
                        else:
                            conv.append([c[i-1], c[i]])
                    if len(conv) < 2 :continue
                    yield conv
        self.data = self.predeal(convs_iter())


