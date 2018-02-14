#!/usr/bin/env python
import glob, os, yaml, sys, re
from nlptools.utils import zload, zdump
from .reader_base import Reader_Base

class Reader_Babi(Reader_Base):
    def __init__(self, cfg):
        Reader_Base.__init__(self, cfg)
     
    def rm_index(self, row):
        return [' '.join(row[0].split(' ')[1:])] + row[1:]

    def read(self, filename):
        cached_pkl = filename + '.pkl'
        if os.path.exists(cached_pkl):
            convs = zload(cached_pkl)
        else:
            convs = []
            with open(filename) as f:
                conv = []
                for l in f:
                    l = l.strip()
                    l = self.rm_index(l.split('\t'))
                    if l[0][:6] == 'resto_':
                        continue
                    if len(l) < 2:
                        if len(conv) > 0 :
                            convs.append(conv)
                            conv = []
                    else:
                        conv.append([l[0], l[1]])
            convs = self.predeal(convs)
            zdump(convs, cached_pkl)
        self.responses.build_mask()
        self.data = convs



