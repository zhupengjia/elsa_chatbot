#!/usr/bin/env python

"""
    Author: Pengjia Zhu (zhupengjia@gmail.com)
"""


class TopicManager:
    def __init__(self):
        """
            Topic Manager to manage skills
        """
        self.skills = {}
        self.current_skill = None

    def register(self, skill_name, skill_instance):
        """
            Register topic

            Input:
                - skill_name: string
                - skill_instance: instance of skill
        """
        self.skills[skill_name] = skill_instance

    def update_response_masks(self, current_status):
        """
            Update response masks after retrieving utterance and before getting response

            Input:
                - current_status: dictionary of status, generated from Dialog_Status module
        """
        current_status["response_mask_" + self.current_skill] = \
            self.skills[self.current_skill].update_mask(current_status)
        return current_status

    def get_topic(self, current_status=None):
        """
            get current topic

            Input:
                - current_status: dictionary of status, generated from dialog_status module. Default is None. If only one skill in topic_manager, will return the only topic name
        """
        if len(self.skills) < 2:
            self.current_skill = list(self.skills.keys())[0]
            return self.current_skill
        self.current_skill = list(self.skills.keys())[0]
        # TODO, Need to implement
        return self.current_skill

    def update_response(self, response_value, current_status):
        """
            update response in the status

            Input:
                - response_value: response target value
                - current_status: dictionary of status, generated from dialog_status module
        """
        current_status["entity"]["RESPONSE"] = None # clear response
        if response_value is not None:
            current_status = self.skills[self.current_skill].update_response(response_value, current_status)
        return current_status

    def get_response(self, current_data, current_status):
        """
            get response from current status

            Input:
                - current_data: data converted from current status
                - current_status: dictionary of status, generated from dialog_status module

        """
        response_value, response_score = self.skills[self.current_skill].get_response(current_data)
        current_status["response_score_"+self.current_skill] = response_score
        if response_value is None:
            return self.get_fallback(current_status)
        response_value = response_value.cpu().detach().numpy()
        response_value = (response_value, response_value>0)
        status = self.update_response(response_value, current_status)
        return status

    def add_response(self, response, current_status):
        """
            add response to current status

            Input:
                - response: string
                - current_status: dictionary of status, generated from dialog_status module

        """
        response_value = self.skills[self.current_skill][response]
        if response_value is None:
            return None
        return self.update_response(response_value, current_status)

    def get_fallback(self, current_status):
        """
            return fallback response
        """
        if self.current_skill is None:
            self.get_topic()
        response_value = self.skills[self.current_skill].get_fallback(current_status)
        if response_value is None:
            return None
        return self.update_response(response_value, current_status)

