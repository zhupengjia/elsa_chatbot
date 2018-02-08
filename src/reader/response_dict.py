#!/usr/bin/env python
import sys, re
from ailab.text import VecTFIDF

class Response_Dict(object):
    def __init__(self, cfg, vocab):
        self.response, self.response_ids = [], []
        self.vocab = vocab
        self.cfg = cfg
        self.__search = VecTFIDF(self.cfg, self.vocab)

    def add(self, response):
        response = re.sub('(\{[A-Z]+\})|(\d+)','', response)
        response_ids = self.vocab.sentence2id(response, ngrams=3) 
        self.response.append(response)
        self.response_ids.append(response_ids)

    def build_index(self):
        self.__search.load_index(self.response_ids)
    
    def __getitem__(self, response):
        response_ids = self.vocab.sentence2id(response)
        if len(response_ids) < 1:
            return None
        result = self.__search.search_index(response_ids)
        if len(result) > 0:
            return result[0][0]
        else:
            return None




