#!/usr/bin/env python
"""
    Author: Pengjia Zhu (zhupengjia@gmail.com)
    Response skill for restapi based chatbot
"""

class RestResponse(SkillBase):
    '''
        Restapi based skill
    '''
    def __init__(self, skill_name, rest_url, **args):
        super().__init__(skill_name)
        self.rest_url = rest_url

    def get_response(self, status_data, current_status):
        pass


