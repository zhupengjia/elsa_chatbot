#!/usr/bin/env python
from bidict import bidict
from nlptools.utils import zload, zdump
import os, numpy

class Entity_Dict:
    def __init__(self, cfg, vocab):
        self.vocab = vocab
        self.cfg = cfg
        self.entity_maskdict = {} #dict for entity mask
        self.load()
    
    #entities map to ids   
    def __call__(self, entities):
        entity_ids  = {}
        for k in entities:
            k_id = self.name2id(k)
            entity_ids[k_id] = []
            for e in entities[k]:
                #value to id
                vid = self.value2id(k_id, e)
                if vid is not None:
                    entity_ids[k_id].append(vid)
        return entity_ids

    #entity name to id
    def name2id(self, entityname):
        if entityname in self.entity_namedict:
            return self.entity_namedict[entityname]
        if len(self.entity_namedict) > 0:
            eid = max(self.entity_namedict.values()) + 1
        else:
            eid = 0
        self.entity_namedict[entityname] = eid
        return eid

    #entity value to id
    def value2id(self, entity_nameid, value):
        if value in self.entity_dict:
            return self.entity_dict[value]
        if entity_nameid in self.entity_type:
            if isinstance(value, (int,float)) and self.entity_type[entity_nameid] != 1:
                return None #skip if k is not a number entity 
            if self.entity_type[entity_nameid] != 0:
                return None #skip if k is not a string entity 
        if len(self.entity_dict) > 0:
            vid = max(self.entity_dict.values()) + 1
        else:
            vid = 0
        if isinstance(value, (int,float)):
            self.entity_value[vid] = value
            self.entity_type[entity_nameid] = 1 #number entity
        else:
            #self.entity_value[vid] = self.vocab.sentence2id(value)
            self.entity_value[vid] = value
            self.entity_type[entity_nameid] = 0 #string entity
        self.entity_dict[value] = vid
        return vid


    def load(self):
        if os.path.exists(self.cfg.entity_dict):
            self.entity_namedict, self.entity_dict, self.entity_value, self.entity_type \
                    = zload(self.cfg.entity_dict)
        else:
            self.entity_namedict = bidict() #entity name to id 
            self.entity_dict = bidict() #entity value to id 
            self.entity_value = {} # entity id to value
            self.entity_type = {} #check if entity is a string(0) or value(1)


    def save(self):
        zdump((self.entity_namedict, self.entity_dict, self.entity_value, self.entity_type), self.cfg.entity_dict)


    def name2onehot(self, entitynameids):
        data = numpy.zeros(self.cfg.max_entity_types, 'float')
        for e in entitynameids:
            if e > self.cfg.max_entity_types - 1:
                continue
            data[e] += 1
        return data






