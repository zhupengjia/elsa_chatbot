#!/usr/bin/env python
"""
    Author: Pengjia Zhu (zhupengjia@gmail.com)
    Response skill for human assistant
"""
from .skill_base import SkillBase
import random

class HumanResponse(SkillBase):
    """
        Human response skill. Must use the same backend as outside backend
    """
    def __init__(self, assistants=None):
        self.assistants = assistants

    def update_response(self, assistant, current_status):
        pass

    def get_response(self, status_data, current_status, **args):
        if self.assistants is None:
            return None, 0
        if not "ASSISTANT" in current_status["entity"]:
            assistant = random.choice(self.assistants)
        else:
            assistant = current_status["entity"]["ASSISTANT"]
        return assistant, 1

