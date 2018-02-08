#!/usr/bin/env python
import glob, os, yaml, sys, re
from nlptools.utils import setLogger, zload, zdump
from .reader_base import Reader_Base

class Reader_Dialog(Reader_Base):
    def __init__(self, cfg):
        Reader_Base.__init__(self, cfg)

    def _read_loop(self, locdir, convs = None):
        #self.logger.info('read files from dir {}'.format(locdir))
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
                        elif l[0] == '<':
                            continue
                        elif l[0] == '=':
                            if len(conv) > 0:
                                convs.append(conv)
                                conv = []
                            continue
                        splitsentence = tuple(re.split(':', l, maxsplit=1))
                        if len(splitsentence) < 1:
                            continue
                        elif len(splitsentence) == 2: 
                            user, sentence = splitsentence
                            sentence = sentence.strip()
                            user = user.strip()
                            if len(user) < 1 or len(sentence) < 1:
                                continue
                            if user.lower() == 'robot':
                                response = sentence
                                conv.append([utterance, response])
                            else:
                                utterance = sentence
                                response = '<SILENCE>'
                        else:
                            continue
                            #conv.append(['<SILENCE>', l.strip()])
                    if len(conv) > 0:
                        convs.append(conv)
                        conv = []
        return convs


    def read(self, locdir):
        cached_pkl = os.path.join(locdir, 'dialog.pkl')
        if os.path.exists(cached_pkl):
            convs = zload(cached_pkl)
        else:
            convs = self._read_loop(locdir)
            convs = self.predeal(convs)
            zdump(convs, cached_pkl)
        #print(len(convs['template']))
        print(convs['response'])
        self.data = convs



