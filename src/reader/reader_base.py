#!/usr/bin/env python
import sys, os, numpy
from nlptools.utils import setLogger
from nlptools.text.ner import NER
from nlptools.text import Vocab, Embedding
from ..module.response_dict import Response_Dict
from ..module.dialog_status import Dialog_Status

class Reader_Base(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.logger = setLogger(self.cfg.logger)
        self.emb = Embedding(self.cfg.ner)
        self.ner = NER(self.cfg.ner)
        self.ner.build_keywords_index(self.emb)
        self.vocab = Vocab(cfg.ner, self.ner, self.emb)
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
                        tokens = [self.vocab.PAD]
                        token_ids = [self.vocab._id_PAD]
                    entity_ids = {}
                    for e in entities:
                        entity_ids[e] = [self.vocab.sentence2id(ee) for ee in entities[e]]
                    if i_s == 0:
                        ripe['utterance'].append(token_ids)
                        ripe['ent_utterance'].append(entity_ids)
                    else:
                        response_id = self.responses[tokens + list(entities.keys())]
                        ripe['ent_response'].append(entity_ids)
                        if response_id is None:
                            ripe['response'].append(0)
                        else:
                            #print('='*60)
                            #print(response_id)
                            #print(' '.join(tokens))
                            #for i in range(min(3, len(response_id))):
                            #    print('-'*30)
                            #    print(self.responses.response[response_id[i][0]])
                            ripe['response'].append(response_id[0])
        self.vocab.reduce_vocab()
        self.vocab.save()
        return ripe

    def __iter__(self):
        for epoch in range(self.cfg.model.epochs):
            dialogs = []
            for n in range(self.cfg.model.batch_size):
                sampleid = numpy.random.randint(len(self.data['idrange']))
                idrange = self.data['idrange'][sampleid]
                data = {'mask':[], 'entities':[]}
                for k in ['utterance', 'response', 'ent_utterance']:
                    data[k] = self.data[k][idrange[0]:idrange[1]]
                #roll entity gets
                dialog_status = Dialog_Status(self.vocab, self.responses.entities)
                for i in range(idrange[0], idrange[1]):
                    #add to dialog status
                    dialog_status.add(self.data['utterance'][i], self.data['response'][i], self.data['ent_utterance'][i], self.responses.func_need[self.data['response'][i]])
                    #mask
                    dialog_status.getmask(self.responses.masks)
                dialogs.append(dialog_status)
            yield Dialog_Status.torch(self.cfg, self.vocab, dialogs, shuffle=True)




