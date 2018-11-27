#!/usr/bin/env python

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

class Behaviors:
    '''
        The build in hook functions
        
        Input should be the current entities, output should be a dictionary, and will be merged to the current entities. The key in dictionary will be the entity name, and the value will be the entity value.  

        Input:
            - cfg: dictionary or nlptools.utils.config object
    '''
    def __init__(self, cfg):
        self.cfg = cfg

    def geteid(self, entities):
        return {'CAREER_LEVEL':'7 - Manager', 'COUNTRY':'China'}
    
    def getwbs(self, entities):
        return {'WBS':'9FBxDMO'}
    
    def getbalance(self, entities):
        return {'BALANCE':10}

    def getrestinfo(self, entities):
        return {'REST_INFO': 'RESERVED_REST_INFO'}

    def bookcuisine(self, entities):
        return {'BOOK_CUISINE': 'DONE!'}

    
