#!/usr/bin/env python

class Generative_Response(Skill_Base):
    '''
        Generative skill for chatbot
    '''

    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer


    def __getitem__(self, response):
        '''
            Predeal response string
        '''
        pass

    
    def get_response(self, current_status):
        '''
            predict response value from current status

            Input:
                - current_status: dictionary of status, generated from Dialog_Status module
        '''
        pass

    def update_response(self, response, current_status):
        '''
            update current response to the response status.
            
            Input:
                - response: value of response
                - current_status: dictionary of status, generated from Dialog_Status module
        '''
        pass

