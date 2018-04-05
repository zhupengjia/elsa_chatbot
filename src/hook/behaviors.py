#!/usr/bin/env python

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

class Behaviors:
    '''
        The build in hook functions
        
        All hook functiosn should be the staticmethod, input should be the current entities, output should be a dictionary, and will be merged to the current entities. The key in dictionary will be the entity name, and the value will be the entity value.
    '''
    @staticmethod
    def geteid(entities):
        return {'CAREER_LEVEL':'7 - Manager', 'COUNTRY':'China'}
    
    @staticmethod
    def getwbs(entities):
        return {'WBS':'9FBxDMO'}
    
    @staticmethod
    def getbalance(entities):
        return {'BALANCE':10}

    @staticmethod
    def getrestinfo(entities):
        return {'REST_INFO': 'RESERVED_REST_INFO'}

    @staticmethod
    def bookcuisine(entities):
        return {'BOOK_CUISINE': 'DONE!'}

    
