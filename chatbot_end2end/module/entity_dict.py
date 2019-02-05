#!/usr/bin/env python
from bidict import bidict
from nlptools.utils import zload, zdump
import os, numpy

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

class Entity_Dict:
    '''
        Entity Dict.

    '''
    
    @staticmethod
    def name2id(entityname, max_entity_types):
        '''
            entity name to id


            Input:
                - entityname: string
                - max_entity_types: int, number of entity types 

            Output:
                - entityname_id: int
        '''
        return hash(entityname)%max_entity_types

    @staticmethod
    def name2onehot(entitynames, max_entity_types):
        '''
            return a entity name list to one-hot numpy array

            Input:
                - entitynames: list of entityname
               
            Output:
                - 1d numpy array
        '''
        data = numpy.zeros(max_entity_types, 'float')
        for e in entitynames:
            eid = Entity_Dict.name2id(e, max_entity_types)
            data[eid] = 1
        return data



