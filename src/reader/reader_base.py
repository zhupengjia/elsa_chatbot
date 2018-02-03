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
        dialogs = []
        for k in ['utterance', 'response', 'ent_utterance', 'ent_response', 'entities']:
            self.data[k] = []
        for dialog in data:
            dialogs.append([])
            for pair in dialog:
                for i, sentence in enumerate(pair):
                    entities, tokens = self.ner.get(sentence.lower())
                    print(entities, tokens)
                    sys.exit()
                    if i == 0:
                        self.data['utterance'].append(tokens)
                        self.data['ent_utterance'].append(entities)
                    else:
                        self.data['response'].append(tokens)
                        self.data['ent_response'].append(entities)



