#!/usr/bin/env python
import os, re, h5py
from .reader_base import Reader_Base

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

class Reader_Babi(Reader_Base):
    '''
        Read from babi training data, inherit from Reader_Base

        Input:
            - see Reader_Base
    '''
    def __init__(self, **args):
        Reader_Base.__init__(self, **args)
     
    def _rm_index(self, row):
        return [' '.join(row[0].split(' ')[1:])] + row[1:]

    def read(self, filepath):
        '''
            Input:
                - filepath: the path of training file
        '''
        cached_data = filepath + '.h5'
        if os.path.exists(cached_data) and os.path.getsize(cached_data) > 10240:
            self.data = h5py.File(cached_data, 'r')
            return

        def convs_iter():
            with open(filepath) as f:
                conv = []
                utterance, response = '', ''
                for l in f:
                    l = l.strip()
                    l = self._rm_index(l.split('\t'))
                    if l[0][:6] == 'resto_':
                        utterance, response = '', ''
                        continue
                    if len(l) < 2:
                        if len(utterance) > 0 and len(response) > 0 :
                            conv.append([utterance, response])
                        if len(conv) > 0 :
                            yield conv
                            conv = []
                        utterance, response = '', ''
                    else:
                        if l[0] == '<SILENCE>' :
                            if l[1][:8] != 'api_call':
                                response += '\n' + l[1]
                        else:
                            if len(utterance) > 0 and len(response) > 0 :
                                conv.append([utterance, response])
                            utterance, response = '', ''
                            utterance = l[0]
                            response = l[1]
                if len(utterance) > 0 and len(response) > 0 :
                    conv.append([utterance, response])
                if len(conv) > 0:
                    yield conv
        
        self.data = self.predeal(convs_iter(), cached_data)


