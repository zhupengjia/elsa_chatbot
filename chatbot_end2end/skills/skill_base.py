#!/usr/bin/env python

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

class Skill_Base:
    '''
        Base skill class. Define some necessaryy methods for a skill
    '''
   
    def update_mask(self, current_status):
        '''
            Update response masks after retrieving utterance and before getting response
            
            Input:
                - current_status: dictionary of status, generated from Dialog_Status module
        '''
        return None

    def get_response(self, current_status):
        '''
            predict response value from current status

            Input:
                - current_status: dictionary of status, generated from Dialog_Status module
        '''
        return None

    def update_response(self, response, current_status)
        '''
            update current response to the response status.
            
            Input:
                - response id: int
                - current_status: dictionary of status, generated from Dialog_Status module
        '''
        return current_status
