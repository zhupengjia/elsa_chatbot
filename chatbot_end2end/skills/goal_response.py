#!/usr/bin/env python
"""
    Author: Pengjia Zhu (zhupengjia@gmail.com)
    Response skill for end2end goal oriented chatbot
"""
import os
import torch
from .rule_response import RuleResponse
from ..model.dialog_tracker import Dialog_Tracker

class GoalResponse(RuleResponse):
    """
        Response skill for goal oriented chatbot. Used to index the response template, get the most closed

        The class will build a tf-idf index for template, the __getitem__ method is to get the most closed response
        via the tf-idf algorithm.(only used for training, the response string in training data will convert to a response id via tfidf search)

        Input:
            - dialogflow: dialogflow instance from ..reader.ReaderXLSX

        Special usage:
            - len(): return number of responses in template
            - __getitem__ : get most closed response id for response, input is response string
    """
    def __init__(self, skill_name, dialogflow, saved_model="dialog_tracker.pt", **args):
        super(GoalResponse, self).__init__(skill_name, dialogflow, **args)
        self.saved_model = saved_model
        self.model = None

    def init_model(self, device='cpu', **args):
        """
            init dialog tracker

            Input:
                - device: string, model location, default is 'cpu'
                - see ..model.dialog_tracker.Dialog_Tracker for more parameters if path of saved_model not existed

        """
        if os.path.exists(self.saved_model):
            checkpoint = torch.load(self.saved_model,
                                    map_location=lambda storage, location: storage)
            self.model = Dialog_Tracker(**{**args, **checkpoint['config_model']})
            self.model.to(device)
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            self.model = Dialog_Tracker(Nresponses=len(self.dialogflow), **args)
            self.model.to(device)

    def eval(self):
        """
        Set model to eval mode
        """
        self.model.eval()

    def get_response(self, status_data):
        """
            predict response value from current status

            Input:
                - status_data: data converted from dialog status
        """
        if self.model.training:
            return self.model(status_data)

        y_prob = self.model(status_data)
        _, y_pred = torch.max(y_prob.data, 1)
        y_pred = int(y_pred.cpu().numpy()[-1])
        return y_pred, y_prob
