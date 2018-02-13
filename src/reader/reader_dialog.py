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
            fn = os.path.join(locdir, fn)
            if os.path.isdir(fn):
                convs = self._read_loop(fn, convs)
            else:
                conv = []
                utterance, response = [], []
                linestatus = 0 #0 robot, 1 user, -1 other 
                with open(fn, 'r') as f:
                    for l in f:
                        l = l.strip()
                        if len(l) < 1 :
                            linestatus = -1
                            continue
                        elif l[0] == '<':
                            linestatus = -1
                            continue
                        elif l[0] == '=':
                            if len(conv) > 0:
                                convs.append(conv)
                                conv = []
                            linestatus = -1
                            continue
                        else: 
                            splitsentence = tuple(re.split(':', l, maxsplit=1))
                            if len(splitsentence) < 1:
                                linestatus = -1
                                continue
                            elif len(splitsentence) == 1:
                                say = l
                            elif len(splitsentence) == 2:
                                user, say = splitsentence
                                user = user.strip()
                                say = say.strip()
                                if user.lower() in ['robot']:
                                    linestatus = 0
                                else:
                                    linestatus = 1
                        if linestatus != 0 and (len(utterance) > 0 or len(response) > 0):
                            if len(utterance) < 1:
                                utterance.append('<SILENCE>')
                            if len(response) > 0:
                                conv.append(['\t'.join(utterance), '\t'.join(response)])
                            utterance, response = [], []

                        if linestatus == 0:
                            response.append(say)
                        elif linestatus == 1:
                            utterance.append(say)

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
        self.responses.build_mask()
        self.data = convs



