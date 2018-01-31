#!/usr/bin/env python
import glob, os, yaml, sys, re
from nlptools.utils import setLogger
from .reader_base import Reader_Base

class Reader_YAML(Reader_Base):
    def __init__(self, cfg):
        Reader_Base.__init__(self, cfg)

    def read(self, yamlfile, bidirectional=False):
        with open(yamlfile, 'r') as f:
            data = yaml.load(f)
        key = ''.join(data['categories'])
        data_raw = data['conversations']
        convs = []
        for c in data_raw: 
            conv = []
            for i in range(0,len(c),2):
                if i+1 < len(c):
                    conv.append([c[i], c[i+1]])
                else:
                    conv.append([c[i], '<SILENCE>'])
            convs.append(conv)
            if bidirectional:
                conv = []
                for i in range(0,len(c),2):
                    if i-1 < 0:
                        conv.append(['<SILENCE>', c[i]])
                    else:
                        conv.append([c[i-1], c[i]])
                if len(conv) < 2 :continue
                convs.append(conv)
        convs = self.predeal(convs)



