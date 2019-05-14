#!/usr/bin/env python
import os, numpy, torch
from nlptools.text.tokenizer import format_sentence
from .skill_base import SkillBase
from ..model.generative_tracker import Generative_Tracker

"""
    Author: Pengjia Zhu (zhupengjia@gmail.com)
"""


class GenerativeResponse(SkillBase):
    """
        Generative skill for chatbot

        Input:
            - tokenizer: instance of nlptools.text.tokenizer
            - vocab:  instance of nlptools.text.vocab
            - max_seq_len: int, maximum sequence length
    """

    def __init__(self, skill_name, tokenizer, vocab,  max_seq_len=100, beam_size=1, **args):
        super(GenerativeResponse, self).__init__(skill_name)
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.beam_size = beam_size

    def __getitem__(self, response):
        """
            Predeal response string
        """
        response_tokens = self.tokenizer(response)
        response_ids = self.vocab.words2id(response_tokens)
        return response_ids

    def init_model(self, saved_model="generative_tracker.pt", device='cpu', **args):
        """
            Initialize model

            Input:
                - saved_model: str, default is "dialog_tracker.pt"
                - device: string, model location, default is 'cpu'
                - see ..model.generative_tracker.Generative_Tracker for more parameters if path of saved_model not existed
        """
        additional_args = {"beam_size": self.beam_size,
                           "skill_name":self.skill_name,
                           "pad_id": self.vocab.PAD_ID,
                           "bos_id": self.vocab.BOS_ID,
                           "eos_id": self.vocab.EOS_ID,
                           "unk_id": self.vocab.UNK_ID,
                           }
        args = {**args, **additional_args}
        if os.path.exists(saved_model):
            self.checkpoint = torch.load(saved_model, map_location=lambda storage, location: storage)
            self.model = Generative_Tracker(**{**args, **self.checkpoint['config_model']}) 
            self.model.to(device)
            self.model.load_state_dict(self.checkpoint['state_dict'])
        else:
            self.model = Generative_Tracker(**args)
            self.model.to(device)

    def eval(self):
        """
        Set model to eval mode
        """
        self.model.eval()

    def get_response(self, status_data):
        """
            predict response value from status

            Input:
                - status_data: data converted from dialog status
        """
        if self.model.training:
            return self.model(status_data)
        else:
            result = self.model(status_data)
            score = numpy.exp(result[1][0])
            return result[0][0], score

    def update_response(self, response, current_status):
        """
            update current response to the response status.

            Input:
                - response: value of response
                - current_status: dictionary of status, generated from Dialog_Status module
        """
        response = response.cpu().detach().numpy()
        current_status["entity"]['RESPONSE'] = self.vocab.id2words(response) 
        response_key = 'response_' + self.skill_name
        mask_key = 'response_mask_' + self.skill_name
        current_status[response_key] = response 

        return current_status

