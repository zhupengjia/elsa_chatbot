#!/usr/bin/env python
import numpy, re
from fuzzywuzzy import fuzz

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

    def get_response(self, status_data, current_status=None, incre_state=None, **args):
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

    def _entity_match(self, regroup, all_entities):
        """
        match and replace entity name in text
        """
        upper = regroup.group(1).upper()
        scores = [fuzz.partial_ratio(upper, e) for e in all_entities]
        max_id = numpy.argmax(scores)
        max_score = scores[max_id]
        if max_score > 80:
            return "{" + all_entities[max_id] +"}"
        else:
            return regroup.group(1)

    def entity_replace(self, text, all_entities):
        """
        match and replace entity name in text
        """
        return re.sub("\{\s*(\w+)\s*\}", lambda x:self._entity_match(x, list(all_entities)), text)

    def get_fallback(self, current_status):
        '''
            Get fallback feedback
        '''
        return ":)", 1

