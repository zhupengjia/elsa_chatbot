#!/usr/bin/env python
import copy

"""
    Author: Pengjia Zhu (zhupengjia@gmail.com)
"""


class TopicManager:
    def __init__(self, config=None):
        """
            Topic Manager to manage skills
        """
        self.skills = {}
        self.skill_names = []
        self.current_skill = None
        self.stradegy = config.strategy if config else "current"

    def register(self, skill_name, skill_instance):
        """
            Register topic

            Input:
                - skill_name: string
                - skill_instance: instance of skill
        """
        self.skills[skill_name] = skill_instance
        self.skill_names.append(skill_name)
        if len(self.skill_names) < 2:
            #init current skill
            self.current_skill = skill_name 

    def update_response_masks(self, current_status):
        """
            Update response masks after retrieving utterance and before getting response

            Input:
                - current_status: dictionary of status, generated from Dialog_Status module
        """
        for s in self.skill_names:
            current_status["response_mask_" + s] = \
                    self.skills[s].update_mask(current_status)
        return current_status

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

    def get_topic(self, current_status=None):
        """
            Todo: will use classification method to choose topic
        """
        return self.current_skill

    def get_response(self, current_data, current_status, incre_state=None):
        """
            get response from current status

            Input:
                - current_data: data converted from current status
                - current_status: dictionary of status, generated from dialog_status module
                - incre_state: incremental state, default is None

        """
        old_skill = self.current_skill
        if self.stradegy == "current" and len(self.skill_names)>1:
            # current skill first
            self.skill_names.remove(self.current_skill)
            self.skill_names.insert(0, self.current_skill)
        for skill in self.skill_names:
            # response mask
            self.current_skill = skill
            response_value, response_score = self.skills[skill].get_response(current_data, current_status, incre_state)
            if response_value is not None:
                break
        if response_value is None:
            self.current_skill = old_skill
            current_status["entity"]["RESPONSE"] = ":)"
            return current_status

        current_status["topic"] = self.current_skill
            
        current_status["response_score_"+self.current_skill] = response_score
        return self.update_response(response_value, current_status)

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

