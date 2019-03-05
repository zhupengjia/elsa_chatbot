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
        self.current_topic = None


    def register(self, topic_name, topic_instance):
        '''
            Register topic
        
            Input:
                - topic_name: string
                - topic_instance: instance of topic
        '''
        self.topics[topic_name] = topic_instance

    
    def update_response_masks(self, current_status):
        '''
            Update response masks after retrieving utterance and before getting response
            
            Input:
                - current_status: dictionary of status, generated from Dialog_Status module
        '''
        current_status["response_mask_" + self.current_topic] = self.topics[self.current_topic].update_mask(current_status)
        return current_status


    def get_topic(self, current_status=None):
        '''
            get current topic
            
            Input:
                - current_status: dictionary of status, generated from dialog_status module. Default is None. If only one skill in topic_manager, will return the only topic name
        '''
        if len(self.topics) < 2:
            self.current_topic = list(self.topics.keys())[0]
            return self.current_topic
        self.current_topic = list(self.topics.keys())[0]
        return self.current_topic


    def update_response(self, response_value, current_status):
        '''
            update response in the status

            Input:
                - response_value: response target value
                - current_status: dictionary of status, generated from dialog_status module
        '''
        current_status = self.topics[self.current_topic].update_response(response_value, current_status)
        return current_status


    def get_response(self, current_status):
        '''
            get response from current status
            
            Input:
                - current_status: dictionary of status, generated from Dialog_Status module

        '''
        response_value = self.topics[self.current_topic].get_response(current_status)
        return self.topics[self.current_topic].update_response(self.current_topic, response_value, current_status)
        
    def add_response(self, response, current_status):
        '''
            add response to current status
            
            Input:
                - response: string
                - current_status: dictionary of status, generated from dialog_status module
    
        '''
        response_value = self.topics[self.current_topic][response]
        return self.topics[self.current_topic].update_response(self.current_topic, response_value, current_status)



