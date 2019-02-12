#!/usr/bin/env python
import numpy, torch, copy, time, sys
from torch import functional as F
from .entity_dict import Entity_Dict
from nlptools.utils import flat_list
from .dialog_data import Dialog_Data
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

def Collate_Fn(batch):
    '''
        Collate function for torch.data.generator
    '''
    N_batch = len(batch)
    if N_batch < 1:
        return []
    data = Dialog_Data({})
    dialog_lengths = numpy.array([b["utterance"].shape[0] for b in batch], "int")
    perm_idx = dialog_lengths.argsort()[::-1].astype("int")
    dialog_lengths = dialog_lengths[perm_idx]
    max_dialog_len = int(dialog_lengths[0])

    def pack_loop(batchdic, targetdic):
        for k in batchdic[0].keys():
            if isinstance(batchdic[0][k], torch.Tensor):
                #padding
                for b in batchdic:
                    padshape = [0]*(b[k].dim()*2)
                    padshape[-1]= int(max_dialog_len - b[k].shape[0])
                    b[k] = F.pad(b[k],padshape)
                #stack
                targetdic[k] = torch.stack([b[k] for b in batchdic])
                #pack
                targetdic[k] = pack_padded_sequence(targetdic[k][perm_idx], dialog_lengths, batch_first=True)
                #extract pack batch
                if not "pack_batch"  in data:
                    data["pack_batch"] = targetdic[k].batch_sizes
                targetdic[k] = targetdic[k].data
            elif isinstance(batchdic[0][k], dict):
                next_loop_batch = [b[k] for b in batchdic]
                targetdic[k] = pack_loop(next_loop_batch, Dialog_Data({}))
        return targetdic
               
    data = pack_loop(batch, data)
   
    return data 


class Dialog_Status:
    def __init__(self, vocab, tokenizer, ner, topic_manager, sentiment_analyzer, max_seq_len=100, max_entity_types=1024, rl_maxloop=20, rl_discount=0.95):
        '''
        Maintain the dialog status in a dialog

        The dialog status will maintain:
            - utterance history
            - response history
            - entity mask history
            - entity history
            - current entity mask
            - current entities
        
        The class also provide the method to convert those data to torch variables
        
        To use this class, one must do the following steps:
            - add_utterance: add the current utterance to status, the entities will be extracted from utterance
            - update_response/get_response: updated the current response to status or get response from current_status

        Input:
            - vocab:  instance of nlptools.text.vocab
            - tokenizer:  instance of nlptools.text.Tokenizer
            - ner: instance of nlptools.text.ner
            - topic_manager: topic manager instance, see src/module/topic_manager
            - sentiment_analyzer: sentiment analyzer instance
            - max_seq_len: int, maximum sequence length
            - max_entity_types: int, maximum entity types

        Special usage:
            - str(): print the current status
            - len(): length of dialog

        '''

        self.tokenizer = tokenizer
        self.ner = ner
        self.vocab = vocab
        self.topic_manager = topic_manager
        self.max_seq_len = max_seq_len
        self.max_entity_types = max_entity_types
        self.rl_maxloop = rl_maxloop
        self.rl_discount = rl_discount

        self.current_status = self.__init_status()

        self.history_status = []
        self.sentiment_analyzer = sentiment_analyzer
        

    def __init_status(self):
        initstatus =  Dialog_Data({"entity":{}, \
                "entity_emb": None, \
                "utterance": None, \
                "utterance_mask": None, \
                "response_string": None, \
                "response": {}, \
                "response_mask": {}, \
                "sentiment": 0, \
                "topic": None \
                })
        return initstatus

    
    @classmethod

    def new_dialog(cls, vocab, tokenizer, ner, topic_manager, sentiment_analyzer, max_seq_len=100, max_entity_types=1024):
        '''
            create a new dialog
        '''
        return cls(vocab, tokenizer, ner, topic_manager, sentiment_analyzer, max_seq_len, max_entity_types)


    def add_utterance(self, utterance):
        '''
            add utterance to status

            Input:
                - utterance: string

            Output:
                - if success, return True, otherwise return None

        '''
        #get entities
        utterance = utterance.strip()
        entities, utterance_replaced = self.ner.get(utterance, return_dict=True)
        for e in entities:
            self.current_status["entity"][e] = entities[e][0] #only keep first value
        self.current_status["entity_emb"] = Entity_Dict.name2onehot(self.current_status["entity"].keys(), self.max_entity_types)
    
        #utterance to id 
        tokens = self.tokenizer(utterance_replaced)
        utterance_ids = self.vocab.words2id(tokens)

        utterance_ids = utterance_ids[:self.max_seq_len-2]
        seq_len = len(utterance_ids) + 2
       
        self.current_status["utterance"] = numpy.zeros(self.max_seq_len, 'int')
        self.current_status["utterance_mask"] = numpy.zeros(self.max_seq_len, 'int')
        
        self.current_status["utterance"][0] = self.vocab.CLS_ID
        self.current_status["utterance"][1:seq_len-1] = utterance_ids
        self.current_status["utterance"][seq_len-1] = self.vocab.SEP_ID
        
        self.current_status["utterance_mask"][:seq_len] = 1
     
        #get topic
        self.current_status["topic"] = self.topic_manager.get_topic(self.current_status)

        #get sentiment
        self.current_status["sentiment"] = self.sentiment_analyzer(utterance)
        
        #response mask
        self.current_status = self.topic_manager.update_response_masks(self.current_status)


    def add_response(self, response):
        '''
            add existed response, usually for training data
            
            Input:
                - response: string
        '''
        response = response.strip()
        entities, response_replaced = self.ner.get(response, return_dict=True)

        self.current_status = self.topic_manager.add_response(response_replaced, self.current_status)
        self.history_status.append(copy.deepcopy(self.current_status))
    
    
    def get_response(self):
        '''
            get response from current status
        '''
        self.current_status = self.topic_manager.get_response(self.current_status)
        self.history_status.append(copy.deepcopy(self.current_status))

    
    def __str__(self):
        '''
            print the current status
        '''
        txt = '='*60 + '\n'
        for k in self.current_status:
            txt += "{}: {}\n".format(k, str(self.current_status[k]))
        return txt


    def __len__(self):
        '''
            length of dialog
        '''
        return len(self.history_status)
   
    
    def data(self, topic_names=None, status_list=None, turn_start = 0):
        '''
            return pytorch data for all messages in this dialog

            Input:
                - topic_names: list of topic name need to return,  default is None to return all available topics
                - status_list: list status need to predeal,default is None for history status
                - turn_start: int, which turn loop for first status, default is 0
        '''
        if status_list is None:
            status_list = self.history_status
        N_status = len(status_list)
        status = Dialog_Data({\
            "entity": numpy.zeros((N_status, self.max_entity_types), 'int'),\
            "utterance": numpy.zeros((N_status, self.max_seq_len), 'int'),\
            "utterance_mask": numpy.zeros((N_status, self.max_seq_len), 'int'),\
            "reward": numpy.zeros(N_status, 'float'), \
            "sentiment": numpy.zeros(N_status, 'float'), \
            "response": {},\
            "response_mask":{} \
        })

        if topic_names is None:
            topic_names = self.topic_manager.topics.keys()
        
        if N_status+turn_start < self.rl_maxloop:
            reward_base = 1
        else:
            reward_base = 0
        
        for i, s in enumerate(status_list):
            status["entity"][i] = s["entity_emb"]
            status["utterance"][i] = s["utterance"]
            status["utterance_mask"][i] = s["utterance_mask"]
            status["sentiment"][i] = s["sentiment"]
            status["reward"][i] = reward_base * self.rl_discount**(i+turn_start)
            for tk in topic_names:
                for k in ["response", "response_mask"]:
                    if tk in s[k]:
                        if not tk in status[k]:
                            if isinstance(s[k][tk], numpy.ndarray):
                                status[k][tk] = numpy.repeat(numpy.expand_dims(numpy.zeros_like(s[k][tk]), axis=0), N_status, axis=0)
                            else:
                                status[k][tk] = numpy.zeros(N_status, type(s[k][tk]))
                        status[k][tk][i] = s[k][tk]
        status["reward"] = status["reward"]/(N_status+turn_start)

        for k in status.keys():
            if k in ["response", "response_mask"]:
                for tk in status[k]:
                    status[k][tk] = torch.tensor(status[k][tk])
            else:
                status[k] = torch.tensor(status[k])
        return status


    def current(self, topic_names=None):
        '''
            return pytorch data for current loop 
        '''
        return self.data(topic_names, [self.current_status], len(self.history_status))


