
#!/usr/bin/env python
import os, h5py, re, sys
from .reader_base import ReaderBase

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

class ReaderUbuntu(ReaderBase):
    '''
        Read from Ubuntu corpus training data, inherit from Reader_Base

        Input:
            - see Reader_Base
    '''
    
    def __init__(self, **args):
        ReaderBase.__init__(self, **args)

    def read(self, filepath):
        for f in os.listdir(filepath):
            fpath = os.path.join(filepath, f)
            if os.path.isdir(fpath):
                self.read(os.path.join(fpath))
            else:
                if os.path.splitext(f)[1] == ".tsv":
                    with open(fpath) as csv:
                        print("="*60)
                        for line in csv:
                            l = line[line.rindex("\t")+1:].strip()
                            print(l)
        ## TODO , not finished

