#!/usr/bin/env python
import numpy

"""
    Author: Pengjia Zhu (zhupengjia@gmail.com)
"""


class EntityDict:
    """
        Entity Dict.

    """
    
    @staticmethod
    def name2id(entity_name, max_entity_types):
        """
            entity name to id


            Input:
                - entityname: string
                - max_entity_types: int, number of entity types 

            Output:
                - entityname_id: int
        """
        return hash(entity_name)%max_entity_types

    @staticmethod
    def name2onehot(entity_names, max_entity_types):
        """
            return a entity name list to one-hot numpy array

            Input:
                - entitynames: list of entityname
               
            Output:
                - 1d numpy array
        """
        data = numpy.zeros(max_entity_types, 'float')
        for e in entity_names:
            eid = EntityDict.name2id(e, max_entity_types)
            data[eid] = 1
        return data



