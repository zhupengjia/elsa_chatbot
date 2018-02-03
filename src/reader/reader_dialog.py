#!/usr/bin/env python
import glob, os, yaml, sys, re
from nlptools.utils import setLogger
from .reader_base import Reader_Base

class Reader_Dialog(Reader_Base):
    def __init__(self, cfg):
        Reader_Base.__init__(self, cfg)

    def _read_loop(self, locdir, convs = None):
        if convs is None:
            convs = []
        for fn in os.listdir(locdir):
            f = os.path.join(locdir, fn)
            if os.path.isdir(f):
                convs = self._read_loop(f, convs)
            else:
                conv = []
                utterance, response = '', ''
                with open(f, 'r') as f:
                    for l in f:
                        l = l.strip()
                        if len(l) < 1 :
                            continue
                        if l[0] == '=':
                            if len(conv) > 0:
                                convs.append(conv)
                                conv = []
                            continue
                        splitsentence = tuple(re.split(':', l, maxsplit=1))
                        if len(splitsentence) < 1:
                            continue
                        elif len(splitsentence) == 2: 
                            user, sentence = splitsentence
                            if user.strip().lower() == 'robot':
                                response = sentence.strip()
                                conv.append([utterance, response])
                            else:
                                utterance = sentence.strip()
                        else:
                            conv.append(['<SILENCE>', l.strip()])
                    if len(conv) > 0:
                        convs.append(conv)
                        print(conv)
                        conv = []
        return convs


    def read(self, locdir):
        convs = self._read_loop(locdir)
        convs = self.predeal(convs)
        return convs



