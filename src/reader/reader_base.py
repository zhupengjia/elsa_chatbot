#!/usr/bin/env python
import sys, os, numpy
from nlptools.utils import setLogger
from nlptools.text.ner import NER
from nlptools.text import Vocab, Embedding
from ..module.response_dict import Response_Dict
from ..module.dialog_status import Dialog_Status
from ..module.entity_dict import Entity_Dict

class Reader_Base(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.logger = setLogger(self.cfg.logger)
        self.emb = Embedding(self.cfg.ner)
        self.ner = NER(self.cfg.ner)
        self.ner.build_keywords_index(self.emb)
        self.vocab = Vocab(cfg.ner, self.ner, self.emb)
        self.vocab.addBE()
        self.entity_dict = Entity_Dict(cfg, self.vocab)
        self.data = {}
    
    #build response index
    def build_responses(self):
        self.responses = Response_Dict(self.cfg.response_template, self.ner, self.entity_dict)
        with open(self.cfg.response_template.data) as f:
            for l in f:
                l = l.strip()
                if len(l) < 1:
                    continue
                self.responses.add(l)
        self.responses.build_index()


    #total number of responses
    def __len__(self):
        return len(self.responses.response)


    #return response string by id
    def get_response(self, responseid, entity=None):
        response = self.responses.response[resposeid]
        if entity is None:
            response = response.format(*entity)
        return response
   
     
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
                    entity_ids = self.entity_dict(entities)
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
        self.emb.save()
        self.entity_dict.save()
        return ripe

    def __iter__(self):
        for epoch in range(self.cfg.model.epochs):
            dialogs = []
            for n in range(self.cfg.model.batch_size):
                sampleid = numpy.random.randint(len(self.data['idrange']))
                idrange = self.data['idrange'][sampleid]
                #roll entity gets
                dialog_status = Dialog_Status(self.vocab, self.entity_dict, self.responses)
                for i in range(idrange[0], idrange[1]):
                    #add to dialog status
                    dialog_status.add(self.data['utterance'][i], self.data['response'][i], self.data['ent_utterance'][i], self.responses.func_need[self.data['response'][i]])
                    #mask
                    dialog_status.getmask(self.responses.masks)
                #print(dialog_status)
                dialogs.append(dialog_status)
            yield Dialog_Status.torch(self.cfg, self.vocab, self.entity_dict, dialogs, shuffle=True)




