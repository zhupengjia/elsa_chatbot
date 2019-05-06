#!/usr/bin/env python

from ..model.rule_based import RuleBased 

"""
    Author: Pengjia Zhu (zhupengjia@gmail.com)
"""

class RuleResponse(SkillBase):
    '''
        Rule based skill
    '''
    def __init__(self, tokenizer, dialog_file, **args):
        pass

    def init_model(self):
        pass

    def update_mask(self, current_status):
        return 0

    def get_response(self, current_status):
        return 0

    def update_response(self, response, current_status):
        return 0

