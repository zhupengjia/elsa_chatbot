#!/usr/bin/env python
import sys
from nlptools.utils import setLogger
from nlptools.text.ner import NER
from nlptools.text import Vocab
from .response_dict import Response_Dict

class Reader_Base(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.logger = setLogger(self.cfg.logger)
        self.ner = NER(self.cfg.ner)
        self.vocab = Vocab(cfg, self.ner)
        self.vocab.addBE()
        self.data = {}
    
    def get_responses(self):
        self.responses = Response_Dict(self.cfg.response_template, self.vocab)
        with open(self.cfg.response_template.data) as f:
            for l in f:
                l = l.strip()
                if len(l) < 1:
                    continue
                self.responses.add(l)
        self.responses.build_index()

    
    def predeal(self, data):
        ripe = {}
        for k in ['utterance', 'response', 'ent_utterance', 'ent_response', 'idrange']:
            ripe[k] = []

        for i_d, dialog in enumerate(data):
            self.logger.info('predeal dialog {}/{}'.format(i_d, len(data)))
            ripe['idrange'].append([len(ripe['utterance']), len(ripe['utterance'])+len(dialog)])
            for i_p, pair in enumerate(dialog):
                for i_s, sentence in enumerate(pair):
                    entities, tokens = self.ner.get(sentence.lower())
                    token_ids = self.vocab.sentence2id(tokens)
                    if len(token_ids) < 1:
                        entities = {}
                        tokens = ['<SILENCE>']
                        token_ids = self.vocab.sentence2id(tokens)
                    if i_s == 0:
                        ripe['utterance'].append(token_ids)
                        ripe['ent_utterance'].append(entities)
                    else:
                        response_id = self.responses[tokens]
                        ripe['ent_response'].append(entities)
                        if response_id is None:
                            ripe['response'].append(0)
                        else:
                            ripe['response'].append(response_id)
        return ripe

