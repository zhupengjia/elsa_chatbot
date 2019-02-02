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
        return self.topics[self.current_topic].update_mask(current_status)


    def get_topic(self, current_status):
        '''
            get current topic
            
            Input:
                - current_status: dictionary of status, generated from dialog_status module
        '''
        return self.topics.keys()[0]


    def update_response(self, response, current_status):
        '''
            update response in the status

            Input:
                - response: response target value
                - current_status: dictionary of status, generated from dialog_status module
        '''
        current_status = self.topics[self.current_topic].update_response(response, current_status)
        current_status["response"][self.current_topic] = response 
        return current_status


    def get_response(self, current_status):
        '''
            get response from current status
            
            Input:
                - current_status: dictionary of status, generated from Dialog_Status module

        '''
        response_id = self.topics[self.current_topic].get_response(current_status)
        return self.update_response(response_id, current_status)
        

