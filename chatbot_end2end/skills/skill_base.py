#!/usr/bin/env python

"""
    Author: Pengjia Zhu (zhupengjia@gmail.com)
"""


class SkillBase:
    """
        Base skill class. Define some necessaryy methods for a skill
    """
    def __init__(self, skill_name):
        self.skill_name = skill_name
   
    def __getitem__(self, response):
        """
            convert response string to value
        """
        return None

    def init_model(self, **args):
        """
            model initialization, used for getting response
        """
        pass

    def eval(self):
        """
            set model to eval mode
        """
        pass

    def update_mask(self, current_status):
        """
            Update response masks after retrieving utterance and before getting response
            
            Input:
                - current_status: dictionary of status, generated from Dialog_Status module
        """
        return 0

    def get_response(self, status_data, current_status=None, incre_state=None):
        """
            predict response value from current status

            Input:
                - status_data: data converted from dialog status
                - current_status: dictionary of status, generated from dialog_status module
                - incre_state: incremental state, default is None
        """
        return None, 0

    def update_response(self, response, current_status):
        """
            update current response to the response status.
            
            Input:
                - response: value of response
                - current_status: dictionary of status, generated from Dialog_Status module
        """
        return current_status

    def get_fallback(self, current_status):
        """
            return fallback response
        """
        current_status["entity"]["RESPONSE"] = None
        return current_status

