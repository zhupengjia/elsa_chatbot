#!/usr/bin/env python
from ..module.topic_manager import Topic_Manager
from nlptools.text.tokenizer import Tokenizer_BERT
from .. import reader as Reader, skills as Skills, hooks as Hooks

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

class InteractSession:
    def __init__(self, topic_manager, device='cpu', logger=None):
        '''
            Flask interact session
            
            Input:
                - topic_manager: instance of topic manager, see ..module.topic_manager
                - device: string of torch device, default is "cpu"
                - logger: logger instance ,default is None
        '''

        super().__init__()
        self.dialog_status = {}

    
    @classmethod
    def build(cls, config):
        '''
            construct session from config

            Input:
                - config: configure dictionary
        '''
        #logger, tokenizer and ner
        logger = setLogger(**config.logger)
        tokenizer = Tokenizer_BERT(**config.tokenizer)
        ner = NER(**config.ner)
        vocab = tokenizer.vocab
        
        #encoder
        encoder = Sentence_Encoder(config.tokenizer.bert_model_name)

        #skills
        topic_manager = Topic_Manager()
        for skill_name in config.skills:
            if not hasattr(Skills, config.skills[skill_name].wrapper):
                raise RuntimeError("Error!! Skill {} not implemented!".format(config.skills[skill_name].wrapper))
            skill_cls = getattr(Skills, config.skills[skill_name].wrapper)
            response = skill_cls(tokenizer=tokenizer, vocab=vocab, hook=hook, max_seq_len=config.model.max_seq_len)
            response.init_model(saved_model=config.skills[skill_name].saved_model, device=config.model.device)
            topic_manager.register(name, response)



