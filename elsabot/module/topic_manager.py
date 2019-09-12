#!/usr/bin/env python
import copy
from ..skills.cmd_response import CMDResponse

"""
    Author: Pengjia Zhu (zhupengjia@gmail.com)
"""


class TopicManager:
    def __init__(self):
        """
            Topic Manager to manage skills
        """
        self.skills = {"cmd": CMDResponse("cmd")}
        self.skill_names = []

    def register(self, skill_name, skill_instance):
        """
            Register topic

            Input:
                - skill_name: string
                - skill_instance: instance of skill
        """
        self.skills[skill_name] = skill_instance
        self.skill_names.append(skill_name)

    def update_response_masks(self, current_status):
        """
            Update response masks after retrieving utterance and before getting response

            Input:
                - current_status: dictionary of status, generated from Dialog_Status module
        """
        for s in current_status["$TOPIC_LIST"]:
            current_status["$TENSOR_RESPONSE_MASK"][s] = \
                    self.skills[s].update_mask(current_status)
        return current_status

    def update_response(self, response_value, current_status):
        """
            update response in the status

            Input:
                - response_value: response target value
                - current_status: dictionary of status, generated from dialog_status module
        """
        current_status["$RESPONSE"] = None # clear response
        if response_value is not None:
            current_status = self.skills[current_status["$TOPIC"]].update_response(response_value, current_status)
        return current_status

    def redirect_message(self, current_status):
        current_status["$RESPONSE"] = current_status["$UTTERANCE"]
        current_status["$SESSION"] = current_status["$REDIRECT_SESSION"]
        current_status["$RESPONSE_SCORE"] = 1
        return current_status

    def get_topic(self, current_status=None):
        """
            Todo: will use classification method to choose topic
        """
        return current_status["$TOPIC"]

    def get_response(self, current_data, current_status, incre_state=None, **args):
        """
            get response from current status

            Input:
                - current_data: data converted from current status
                - current_status: dictionary of status, generated from dialog_status module
                - incre_state: incremental state, default is None

        """
        old_skill = current_status["$TOPIC"]
        skill_names = copy.deepcopy(current_status["$TOPIC_LIST"])
        origin_skill_names = copy.deepcopy(current_status["$TOPIC_LIST"])
        
        finished_skill = set()

        # CMD skill
        response_value, response_score = self.skills["cmd"].get_response(current_data, current_status)
        if response_value is not None:
            current_status = self.skills["cmd"].update_response(response_value, current_status)
            if current_status["$RESPONSE"] is not None:
                return current_status

        # check if redirect message
        if current_status["$REDIRECT_SESSION"]:
            current_status = self.redirect_message(current_status)
            return current_status

        while len(skill_names) > 0:
            skill = skill_names[0]
            if current_status["$TOPIC_LIST"] != origin_skill_names:
                skill_names = copy.deepcopy(current_status["$TOPIC_LIST"])
                origin_skill_names = copy.deepcopy(current_status["$TOPIC_LIST"])
                continue
            if skill in finished_skill:
                skill_names.pop(0)
                continue
            if current_status["$TOPIC_NEXT"] and skill != current_status["$TOPIC_NEXT"]:
                #check if TOPIC_NEXT in entity, if yes, will try to jump to that skill
                finished_skill.add(skill)
                skill_names.pop(0)
                continue
            current_status["$TOPIC"] = skill
            # response mask
            response_value, response_score = self.skills[skill].get_response(current_data, current_status, incre_state=incre_state, **args)
            current_status["$TOPIC_NEXT"] = None #clear TOPIC_NEXT
            current_status["$RESPONSE_SCORE"] = response_score
            finished_skill.add(skill)
            skill_names.pop(0)
            if response_value is not None:
                # get final response
                current_status = self.update_response(response_value, current_status)
                if current_status["$RESPONSE"] is not None:
                    break
        if response_value is None:
            current_status["$TOPIC"] = old_skill
            current_status["$RESPONSE"] = ":)"
            current_status["$RESPONSE_SCORE"] = 0
            return current_status

        return current_status

    def add_response(self, response, current_status):
        """
            add response to current status

            Input:
                - response: string
                - current_status: dictionary of status, generated from dialog_status module

        """
        response_value = self.skills[current_status["$TOPIC"]][response]
        if response_value is None:
            return None
        return self.update_response(response_value, current_status)

    def get_fallback(self, current_status):
        response_value, response_score = self.skills[current_status["$TOPIC"]].get_fallback(current_status)
        if response_value is None:
            return None
        return self.update_response(response_value, current_status)


