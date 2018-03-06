#!/usr/bin/env python
import sys, re, numpy
from ailab.text import VecTFIDF, Vocab
from ailab.utils import flat_list

class Response_Dict(object):
    def __init__(self, cfg, tokenizer, entity_dict):
        self.response, self.response_ids, self.func_need = [], [], []
        self.entity_need = {'need':[], 'notneed':[]}
        self.vocab = Vocab(cfg, tokenizer) #response vocab, only used for searching best matched response template, independent with outside vocab.  
        self.cfg = cfg
        self.entity_dict = entity_dict
        self.__search = VecTFIDF(self.cfg, self.vocab)


    def add(self, response):
        response = [x.strip() for x in response.split('|')]
        if len(response) < 3:
            return
        entity_need = {}
        try:
            entity_need['need'], entity_need['notneed'], func_need, response = tuple(response)
        except Exception as err:
            print("Error: Template error!!")
            print("Error sentence: " + ' | '.join(response))
            print("The format should be: needentity | notneedentity | func | response")
            sys.exit()

        response = re.sub('(\{[A-Z]+\})|(\d+)','', response)
        response_ids = self.vocab.sentence2id(response) 
        if len(response_ids) < 1:
            return
        self.response.append(response)
        self.response_ids.append(response_ids)
        
        entity_need = {k:[x.strip() for x in re.split(',', entity_need[k])] for k in entity_need}
        entity_need = {k:[x.upper() for x in entity_need[k] if len(x) > 0] for k in entity_need}
        entity_need = {k:[self.entity_dict.name2id(x) for x in entity_need[k]] for k in entity_need}
        for k in self.entity_need: self.entity_need[k].append(entity_need[k])
        
        func_need = [x.strip() for x in re.split(',', func_need)]
        func_need = [x.lower() for x in func_need if len(x) > 0]
        self.func_need.append(func_need)


    #build search index for response template
    def build_index(self):
        self.__search.load_index(self.response_ids)
        self.vocab.save()


    #build entity mask of response template
    def build_mask(self):
        entity_maskdict = sorted(list(set(flat_list(flat_list(self.entity_need.values())))))
        entity_maskdict = dict(zip(entity_maskdict, range(len(entity_maskdict))))
        self.masks = {'need': numpy.zeros((len(self.response), len(entity_maskdict)), 'bool_'), \
                'notneed': numpy.zeros((len(self.response), len(entity_maskdict)), 'bool_')}
        self.entity_dict.entity_maskdict = entity_maskdict #copy maskdict to entity_dict 
        for i in range(len(self.entity_need['need'])):
            for e in self.entity_need['need'][i]:
                self.masks['need'][i, entity_maskdict[e]] = True
        for i in range(len(self.entity_need['notneed'])):
            for e in self.entity_need['notneed'][i]:
                self.masks['notneed'][i, entity_maskdict[e]] = True
   

    #get most closed response id from templates
    def __getitem__(self, response):
        response_ids = self.vocab.sentence2id(response)
        if len(response_ids) < 1:
            return None
        result = self.__search.search_index(response_ids, topN=1)
        if len(result) > 0:
            return result[0]
        else:
            return None



