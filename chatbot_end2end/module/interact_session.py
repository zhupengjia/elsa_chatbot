#!/usr/bin/env python
from ..module.topic_manager import Topic_Manager

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

class InteractSession:
    '''
        Flask interact session
    '''
    def __init__(self):
        super().__init__()
        self.dialog_status = {}

    
    @classmethod
    def build(cls, skills):
        pass


