#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import numpy, pandas, re, random, math, sys
from ailab.text import Vocab, Embedding, Segment, DocSim
from ..reader.rulebased import Reader

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

class Rule_Based:
    '''
        Rule based chatbot

        Input:
            - cfg: dictionary or ailab.utils.config object
                - needed keys:
                    - all needed keys in ailab.text.segment, embedding and vocab
            - hook: hook instance, please check src/hook/babi_gensays.py for example
    '''
    def __init__(self, cfg, hook):
        self.cfg = cfg
        self.hook = hook
        self.tokenizer = Segment(self.cfg)
        self.embedding = Embedding(self.cfg)
        self.vocab = Vocab(self.cfg, self.tokenizer, self.embedding)
        self.vocab.addBE()
        self.docsim = DocSim(self.vocab)
        self.reader = Reader(self.cfg)
        self.session = {}
        self._predeal()


    def _predeal(self):
        '''
            predeal the dialogs
        '''
        def utterance2id(utter):
            if not isinstance(utter, str):
                return None
            utter = [self.vocab.sentence2id(u) for u in re.split('\n', utter)]
            utter = [u for u in utter if len(u) > 0]
            if len(utter) < 1: return None
            return utter
        self.reader.data['utterance'] = self.reader.data['userSays'].apply(utterance2id)


    def get_reply(self, utterance, clientid):
        '''
            get response from utterance

            Input:
                - utterance: string
                - entities: dictionary
        '''
        #special command
        utterance = utterance.strip()
        if utterance in [':CLEAR', ':RESET', ':RESTART', ":EXIT", ":STOP", ":QUIT", ":Q"]:
            self.reset(clientid)
            return 'dialog status reset!'
        
        #create new session for user
        if not clientid in self.session:
            self.session[clientid] = {'CHILDID': None, 'RESPONSE': None}

        utterance_id = self.vocab.sentence2id(utterance)
        self.session[clientid]['RESPONSE'] = None # clean response
        if len(utterance_id) < 1:
            for e, v in self._get_fallback(self.session[clientid]).items():
                self.session[clientid][e] = v
        else:
            for e, v in self._find(utterance_id, self.session[clientid]).items():
                self.session[clientid][e] = v
        if isinstance(self.session[clientid]['RESPONSE'], str):
            return self.session[clientid]['RESPONSE']
        return '^_^'
    

    def reset(self, clientid):
        '''
            reset session
        '''
        if clientid in self.session:
            del self.session[clientid]


    def _find(self, utterance_id, entities):
        '''
            find the most closed one
            
            Input:
                - utterance_id : utterance token id list
                - entities: dictionary, current entities
        '''
        def getscore(utter_cand):
            if utter_cand is None: 
                return 0
            distance = min([self.docsim.rwmd_distance(utterance_id, u) for u in utter_cand])
            return 1/(1+distance)
        data = self.reader.data
        
        entities['CHILDID'] = [4,6,2]
        if entities['CHILDID'] is not None:
            data_filter = data.loc[entities['CHILDID']]
            data_filter['score'] = data_filter['utterance'].apply(getscore)
            idx = data_filter['score'].idxmax()
            if data_filter.loc[idx]['score'] < self.cfg['min_score']:
                #try to get score for out of rules
                otherids = list(set(list(data.index)) - set(entities['CHILDID']))
                data_others = data.loc[otherids]
                data_others['score'] = data_others['utterance'].apply(getscore)
                idx_others = data_others['score'].idxmax()
                if data_others.loc[idx_others]['score'] > data_filter.loc[idx]['score']:
                    idx = idx_others
        else:
            data['score'] = data['utterance'].apply(getscore)
            idx = data['score'].idxmax()
        return self._get_response(data.loc[[idx]], entities)


    def _get_fallback(self, entities):
        '''
            get fallback, if there is NaN in userSays then pick one of them 

            Input:
                - entities: dictionary, current entities
        '''
        data = self.reader.data[pandas.isnull(self.reader.data.userSays)]
        if len(data) < 1:
            return {}
        return self._get_response(data, entities)


    def _get_response(self, data, entities):
        '''
            get response from data

            Input:
                - data: dataframe
                - entities: dictionary, current entities
        '''
        data = data.loc[random.choice(data.index)] # random pickup one
        if isinstance(data.webhook, str):
            entities = self._call_hook(data.webhook, entities)
        if not isinstance(entities['RESPONSE'], str) and isinstance(data.response, str):
            entities['RESPONSE'] = data.response
        if data.childID is not None:
            entities['CHILDID'] = data.childID
        else:
            entities['CHILDID'] = None
        return entities

        
    def _call_hook(self, hooks, entities):
        hooks = [y for y in [x.strip() for x in re.split('\s,;', hooks)] if len(y) > 0]
        for hook in hooks:
            func = getattr(self.hook, hook)
            if func is not None:
                for e, v in func(entities).items():
                    entities[e] = v
        return entities
        


            
         


