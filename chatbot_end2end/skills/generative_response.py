#!/usr/bin/env python
import os, numpy, torch
from nlptools.text.tokenizer import format_sentence
from .skill_base import SkillBase
from ..model.generative_tracker import GenerativeTracker

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
        return format_sentence(response, vocab=self.vocab,
                                 tokenizer=self.tokenizer, max_seq_len=self.max_seq_len)

    def init_model(self, saved_model="generative_tracker.pt", device='cpu', **args):
        """
            Initialize model

            Input:
                - saved_model: str, default is "generative_tracker.pt"
                - device: string, model location, default is 'cpu'
                - see ..model.generative_tracker.GenerativeTracker for more parameters if path of saved_model not existed
        """
        additional_args = {"beam_size": self.beam_size,
                           "skill_name":self.skill_name,
                           "vocab_size": self.vocab.vocab_size,
                           "max_seq_len": self.max_seq_len,
                           "pad_id": self.vocab.PAD_ID,
                           "bos_id": self.vocab.BOS_ID,
                           "eos_id": self.vocab.EOS_ID,
                           "unk_id": self.vocab.UNK_ID,
                           }
        args = {**args, **additional_args}
        if os.path.exists(saved_model):
            self.checkpoint = torch.load(saved_model, map_location=lambda storage, location: storage)
            model_cfg = self.checkpoint['config_model']
            def copy_args(target_key, source_layer, source_key):
                if source_key in model_cfg[source_layer]:
                    args[target_key] = model_cfg[source_layer][source_key]

            copy_args("vocab_size", "encoder", "vocab_size")
            copy_args("encoder_hidden_layers", "encoder", "num_hidden_layers")
            copy_args("encoder_hidden_size", "encoder", "hidden_size")
            copy_args("encoder_intermediate_size", "encoder", "intermediate_size")
            copy_args("encoder_attention_heads", "encoder", "num_attention_heads")
            copy_args("max_position_embeddings", "encoder", "max_position_embeddings")
            copy_args("decoder_hidden_layers", "decoder", "num_hidden_layers")
            copy_args("decoder_attention_heads", "decoder", "num_attention_heads")
            copy_args("decoder_hidden_size", "decoder", "intermediate_size")
            copy_args("shared_embed", "decoder", "shared_embed")
            self.model = GenerativeTracker(**args) 
            self.model.to(device)
            print("load model from file {}".format(saved_model))
            self.model.load_state_dict(self.checkpoint['state_dict'])
        else:
            self.model = GenerativeTracker(**args)
            print("Create new model")
            self.model.to(device)

    def eval(self):
        """
        Set model to eval mode
        """
        self.model.eval()

    def get_response(self, status_data, incre_state=None):
        """
            predict response value from status

            Input:
                - status_data: data converted from dialog status
                - incre_state: incremental state, default is None
        """
        if self.model.training:
            return self.model(status_data)
        else:
            result = self.model(status_data, incre_state)
            score = numpy.exp(result[1][0])
            response_value = result[0][0].cpu().detach().numpy()
            return response_value, score

    def update_response(self, response, current_status):
        """
            update current response to the response status.

            Input:
                - response: value of response
                - current_status: dictionary of status, generated from Dialog_Status module
        """

        if isinstance(response, tuple):
            response, response_mask = response
        else:
            response_mask = response > 0
        current_status["entity"]['RESPONSE'] = self.tokenizer.tokens2sentence(self.vocab.id2words(response[response_mask.astype("bool_")][1:-1]))

        response_key = 'response_' + self.skill_name
        response_mask_key = 'response_mask_' + self.skill_name
        current_status[response_key] = response
        current_status[response_mask_key] = response_mask

        return current_status

