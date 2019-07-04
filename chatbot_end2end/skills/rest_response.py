#!/usr/bin/env python
import requests, json, random
from .skill_base import SkillBase
"""
    Author: Pengjia Zhu (zhupengjia@gmail.com)
    Response skill for restapi based chatbot
"""

class RestResponse(SkillBase):
    '''
        Restapi based skill
    '''
    def __init__(self, skill_name, rest_url, timeout=3, **args):
        super().__init__(skill_name)
        self.rest_url = rest_url
        self.session_id = str(random.randint(10000000,99999999))
        self.timeout = timeout
        self.headers = {'User-Agent' : "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/534.30 (KHTML, like Gecko) Ubuntu/11.04 Chromium/12.0.742.112 Chrome/12.0.742.112 Safari/534.30"}
        self.rest_post("clear")

    def rest_post(self, sentence):
        data = {
            'text': sentence,
            'session': self.session_id
        }
        try:
            data = requests.post(url=self.rest_url,
                                 data=data,
                                 headers=self.headers,
                                 timeout=self.timeout)
            return data.json()["data"]
        except:
            return {"response": None, "score": 0}

    def update_response(self, response, current_status):
        current_status['response_' + self.skill_name] = response
        current_status["entity"]["RESPONSE"] = response
        return current_status

    def get_response(self, status_data, current_status, incre_state=None):
        result = self.rest_post(current_status["entity"]["UTTERANCE"])
        return result["response"], result["score"]
