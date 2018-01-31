#!/usr/bin/env python
import sys
from nlptools.utils import setLogger
from ..ner.ner_api import Entity

class Reader_Base(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.logger = setLogger(self.cfg.logger)
        self.ner = Entity(self.cfg.ner)
        self.data = {}
    
    def predeal(self, data):
        for dialog in data:
            for pair in dialog:
                for sentence in pair:
                    entities, tokens = self.ner.get(sentence.lower())
                    print(tokens)    

