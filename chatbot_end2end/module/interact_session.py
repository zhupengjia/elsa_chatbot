#!/usr/bin/env python
from ..module.topic_manager import Topic_Manager
from nlptools.text.tokenizer import Tokenizer_BERT

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

class InteractSession:
    '''
        Flask interact session
    '''
    def __init__(self):
        super().__init__()
        self.dialog_status = {}

    
    @classmethod
    def build(cls, config):
        #logger, tokenizer and ner
        logger = setLogger(**config.logger)
        tokenizer = Tokenizer_BERT(**config.tokenizer)
        ner = NER(**config.ner)
        vocab = tokenizer.vocab

        #skills
        topic_manager = Topic_Manager()
        for skill_name in config.skills:
            response = skill_cls(tokenizer=tokenizer, vocab=vocab, hook=hook, max_seq_len=config.reader.max_seq_len, **config.skill)
            topic_manager.register(name, )

    
