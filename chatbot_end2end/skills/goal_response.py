#!/usr/bin/env python
"""
    Author: Pengjia Zhu (zhupengjia@gmail.com)
    Response skill for end2end goal oriented chatbot
"""
import os, torch, numpy
from .rule_response import RuleResponse
from ..model.dialog_tracker import DialogTracker

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
    def __init__(self, skill_name, dialogflow, **args):
        super(GoalResponse, self).__init__(skill_name, dialogflow, **args)
        self.model = None

    def init_model(self, saved_model="dialog_tracker.pt", device='cpu', **args):
        """
            init dialog tracker

            Input:
                - saved_model: str, default is "dialog_tracker.pt"
                - device: string, model location, default is 'cpu'
                - see ..model.dialog_tracker.DialogTracker for more parameters if path of saved_model not existed

        """
        additional_args = {"skill_name":self.skill_name}
        args = {**args, **additional_args}
        if os.path.exists(saved_model):
            self.checkpoint = torch.load(saved_model,
                                    map_location=lambda storage, location: storage)

            model_cfg = self.checkpoint['config_model']
            def copy_args(target_key, source_layer, source_key):
                if source_key in model_cfg[source_layer]:
                    args[target_key] = model_cfg[source_layer][source_key]
            copy_args("vocab_size", "encoder", "vocab_size")
            copy_args("encoder_hidden_layers", "encoder", "num_hidden_layers")
            copy_args("encoder_hidden_size", "encoder", "hidden_size")
            copy_args("encoder_intermediate_size", "encoder", "intermediate_size")
            copy_args("encoder_attention_heads", "encoder", "num_attention_heads")
            copy_args("entity_layers", "decoder", "entity_layers")
            copy_args("entity_emb_dim", "decoder", "entity_emb_dim")
            copy_args("num_hidden_layers", "decoder", "num_hidden_layers")
            copy_args("hidden_size", "decoder", "hidden_size")
            copy_args("num_responses", "decoder", "num_responses")
            copy_args("max_entity_types", "decoder", "max_entity_types")
            self.model = DialogTracker(**args)
            
            self.model.to(device)
            self.model.load_state_dict(self.checkpoint['state_dict'])
        else:
            self.model = DialogTracker(num_responses=len(self.dialogflow), **args)
            self.model.to(device)

    def eval(self):
        """
        Set model to eval mode
        """
        self.model.eval()

    def get_response(self, status_data, incre_state=None):
        """
            predict response value from current status

            Input:
                - status_data: data converted from dialog status
                - incre_state: incremental state, default is None
        """
        if self.model.training:
            return self.model(status_data)

        y_prob = self.model(status_data, incre_state)
        _, y_pred = torch.max(y_prob.data, 1)
        y_pred = int(y_pred.cpu().numpy()[-1])
        score = y_prob[0, y_pred].cpu().detach().numpy()
        score = numpy.exp(score)
        return y_pred, score
