#!/usr/bin/env python
import numpy, torch, copy, time, sys
from torch import functional as F
from .entity_dict import Entity_Dict
from nlptools.utils import flat_list
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

class Dialog_Status:
    def __init__(self, vocab, tokenizer, ner, topic_manager, sentiment_analyzer, max_seq_len=100):
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

        Special usage:
            - str(): print the current status
            - len(): length of dialog

        '''

        self.tokenizer = tokenizer
        self.ner = ner
        self.vocab = vocab
        self.topic_manager = topic_manager
        self.max_seq_len = max_seq_len

        self.current_status = self.__init_status()

        self.history_status = []
        self.sentiment_analyzer = sentiment_analyzer
        

    def __init_status(self):
        initstatus =  {"entity":{}, \
                "utterance": None, \
                "utterance_mask": None, \
                "response_string": None, \
                "response": {}, \
                "response_masks": {}, \
                "sentiment": 0, \
                "topic": None \
                }
        return initstatus

    
    @class_method
    def new_dialog(cls, vocab, tokenizer, ner, topic_manager, sentiment_analyzer, max_seq_len=100):
        '''
            create a new dialog
        '''
        return cls(vocab, tokenizer, ner, topic_manager, sentiment_analyzer, max_seq_len)


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

        #utterance to id 
        tokens = self.tokenizer(utterance_replaced)
        utterance_ids = self.vocab.words2id(tokens)

        utterance_ids = utterance_ids[:self.max_seq_len-2]
        seq_len = len(utterance_ids) + 2
       
        self.current_status["utterance"] = numpy.zeros(self.max_seq_len)
        self.current_status["utterance_mask"] = numpy.zeros(self.max_seq_len)
        
        self.current_status["utterance"][0] = self.vocab.CLS_ID
        self.current_status["utterance"][1:seq_len-1] = utterance_ids
        self.current_status["utterance"][seq_len-1] = self.vocab.SEP_ID
        
        self.current_status["utterance_mask"][:seq_len] = 1
     
        #get topic
        self.current_status["topic"] = self.topic_manager.get_topic(self.current_status)

        #get sentiment
        self.current_status["sentiment"] = self.sentiment_analyzer(utterance)
        
        #response mask
        self.current_status["response_mask"] = self.topic_manager.update_response_masks(self.current_status)


    def add_response(self, response):
        '''
            add existed response
            
            Input:
                - response: string
        '''
        response = response.strip()
        entities, response_replaced = self.ner.get(response, return_dict=True)

        self.current_status = self.topic_manager.add_response(response_replaced)
        self.history_status.append(copy.deepcopy(self.current_status))
    
    
    def get_response(self, response):
        '''
            get response from current status
            
            Input:
                - response: response target value
        '''
        self.current_status = self.topic_manager.get_response(response)
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


    @staticmethod 
    def torch(entity_dict, dialogs, max_entity_types, rl_maxloop=20, rl_discount=0.95, device=None):
        '''
            staticmethod, convert dialogs to batch

            Input:
                - entity_dict: src/module/entity_dict instance
                - dialogs: list of src/module/dialog_status instance
                - max_entity_types: int, number of entity types
                - rl_maxloop: int, maximum dialog loop, default is 20
                - rl_discount: float, discount rate for reinforcement learning, default is 0.95
                - device: instance of torch.device, default is cpu device

            Output:
                - dictionary of pytorch variables with the keys of :
                    - utterance: 2d long tensor
                    - attention_mask: 2d long tensor for input mask
                    - response: 1d long tensor
                    - response_prev: 2d float tensor
                    - mask: 2d float tensor for response mask
                    - entity: 2d float tensor
                    - reward: 1d float tensor, used in policy gradiant
                    - batch_sizes: used in lstm pack_padded_sequence to speed up
        '''
        dialog_lengths = numpy.array([len(d.utterances) for d in dialogs], 'int')
        perm_idx = dialog_lengths.argsort()[::-1]
        N_dialogs = len(dialogs)
        dialog_lengths = dialog_lengths[perm_idx]
        max_dialog_len = int(dialog_lengths[0])

        max_seq_len = len(dialogs[0].utterances[0])
        
        utterance = numpy.zeros((N_dialogs, max_dialog_len, max_seq_len), 'int')
        attention_mask = numpy.zeros((N_dialogs, max_dialog_len, max_seq_len), 'int')
        response = numpy.zeros((N_dialogs, max_dialog_len), 'int')
        entity = numpy.zeros((N_dialogs, max_dialog_len, max_entity_types), 'float')
        mask = numpy.zeros((N_dialogs, max_dialog_len, len(dialogs[0].masks[0])), 'float')
        reward = numpy.zeros((N_dialogs, max_dialog_len), 'float')
        response_prev = numpy.zeros((N_dialogs, max_dialog_len, len(dialogs[0].response_dict)), 'float') #previous response, onehot

        if device is None:
            device = torch.device('cpu')

        for j, idx in enumerate(perm_idx):
            dialog = dialogs[idx]
            #check if dialog finished by measuring the length of dialog
            if len(dialog) < rl_maxloop:
                reward_base = 1
            else:
                reward_base = 0
            response[j, :len(dialog.responses)] = torch.LongTensor(dialog.responses)
            for i in range(dialog_lengths[j]):
                entities = entity_dict.name2onehot(dialog.entities[i].keys(), max_entity_types) #all entity names
                utterance[j, i] = dialog.utterances[i]
                attention_mask[j, i] = dialog.utterance_masks[i]
                entity[j, i, :] = entities
                reward[j, i] = reward_base * rl_discount ** i
                mask[j, i, :] = dialog.masks[i].astype('float')
                if i > 0:
                    response_prev[j, i, response[j, i-1]] = 1
            reward[j] = reward[j]/dialog_lengths[j]
       

        #Get baseline of reward, an average return of the current policy, then do R-b
        for r in numpy.unique(response):
            rid = response == r
            reward_mean = numpy.mean(reward[rid])
            reward[rid] = reward[rid] - reward_mean


        #to torch tensor
        utterance = torch.LongTensor(utterance).to(device)
        attention_mask = torch.LongTensor(attention_mask).to(device)
        response = torch.LongTensor(response).to(device)
        entity = torch.FloatTensor(entity).to(device)
        mask = torch.FloatTensor(mask).to(device)
        reward = torch.FloatTensor(reward).to(device)
        response_prev = torch.FloatTensor(response_prev).to(device)


        #pack them up for different dialogs
        utterance = pack_padded_sequence(utterance, dialog_lengths, batch_first=True)
        attention_mask = pack_padded_sequence(attention_mask, dialog_lengths, batch_first=True).data
        response = pack_padded_sequence(response, dialog_lengths, batch_first=True).data
        reward = pack_padded_sequence(reward, dialog_lengths, batch_first=True).data
        entity = pack_padded_sequence(entity, dialog_lengths, batch_first=True).data
        mask = pack_padded_sequence(mask, dialog_lengths, batch_first=True).data
        response_prev = pack_padded_sequence(response_prev, dialog_lengths, batch_first=True).data
        batch_sizes = utterance.batch_sizes
        utterance = utterance.data

        dialog_torch = {'utterance': utterance, 'attention_mask':attention_mask, 'response':response, 'mask':mask, 'entity':entity, 'reward':reward, 'batch_sizes':batch_sizes}
        return dialog_torch



