#!/usr/bin/env python
from bidict import bidict
from nlptools.utils import zload, zdump
import os, numpy

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

class Entity_Dict:
    '''
        Entity dictionary, used to convert entity names and values to ids.

        Two dictionaries are in this dict:  
            - entity name to id  
            - entity value to id  

        There are also maintained a entity_mask dict, which is used to convert existed entities to a one-hot array

        Input:
            - max_entity_types: int, number of entity types 
            - entity_dict_path: string, cached dict file path, will create a new empty dict if the file not exists. Default is None


        Special usages:
            - __call__: convert entity from entity map like {entityname:[entityvalues],...} to {entityname_id:[entityvalue_ids], ...}

    '''
    def __init__(self, max_entity_types, entity_dict_path=None):
        self.entity_dict_path = entity_dict_path
        self.max_entity_types = max_entity_types
        self.entity_maskdict = {} #dict for entity mask
        self.load()
    
    def __call__(self, entities):
        '''
            entities map to ids   
        '''
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

    def name2id(self, entityname):
        '''
            entity name to id

            Input:
                - entityname: string

            Output:
                - entityname_id: int
        '''
        if entityname in self.entity_namedict:
            return self.entity_namedict[entityname]
        if len(self.entity_namedict) > 0:
            eid = max(self.entity_namedict.values()) + 1
        else:
            eid = 0
        self.entity_namedict[entityname] = eid
        return eid


    def value2id(self, entity_nameid, value):
        '''
            entity value to id

            Input:
                - entity_nameid: int, the entityname_id
                - value: any types. the entity value

            Output:
                - value_id: int
        '''
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


    def load(self, entity_dict_path=None):
        '''
            load the entity dictionary from file, if not exists, generate a new one

            Input:
                - entity_dict_path: string, cached dict file path, will use the global variable if it is None. Default is None.
        '''
        if entity_dict_path is not None:
            self.entity_dict_path = entity_dict_path
        if self.entity_dict_path is not None and os.path.exists(self.entity_dict_path):
            self.entity_namedict, self.entity_dict, self.entity_value, self.entity_type \
                    = zload(self.entity_dict_path)
        else:
            self.entity_namedict = bidict() #entity name to id 
            self.entity_dict = bidict() #entity value to id 
            self.entity_value = {} # entity id to value
            self.entity_type = {} #check if entity is a string(0) or value(1)


    def save(self, entity_dict_path=None):
        '''
            save the entity dictionary to file

            Input:
                - entity_dict_path: string, cached dict file path, will use the global variable it is None. Default is None.
        '''
        if entity_dict_path is not None:
            self.entity_dict_path = entity_dict_path
        if self.entity_dict_path is not None:
            zdump((self.entity_namedict, self.entity_dict, self.entity_value, self.entity_type), self.entity_dict_path)


    def name2onehot(self, entitynameids):
        '''
            return a entity name list to one-hot numpy array

            Input:
                - entitynameids: list of entityname_id
               
            Output:
                - 1d numpy array
        '''
        data = numpy.zeros(self.max_entity_types, 'float')
        for e in entitynameids:
            if e > self.max_entity_types - 1:
                continue
            data[e] += 1
        return data






