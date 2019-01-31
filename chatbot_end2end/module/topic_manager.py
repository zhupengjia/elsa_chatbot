#!/usr/bin/env python


'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

class Topic_Manager:
    def __init__(self):
        '''
            Topic Manager
        '''
        self.topics = {}


    def register(self, topic_name, topic_instance):
        '''
            Register topic
        '''
        self.topics["topic_name"] = topic_instance


    

