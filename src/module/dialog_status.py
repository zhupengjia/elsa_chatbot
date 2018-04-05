#!/usr/bin/env python
import numpy, torch, copy, time, sys
from torch.autograd import Variable
from torch import functional as F
from ..hook import *
from ailab.utils import flat_list
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

class Dialog_Status:
    def __init__(self, cfg, vocab, entity_dict, response_dict):
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
            - getmask: create entity mask from existed entities
            - add_response: add the current response to status, and run the corresponded hook functions

        Input:
            - cfg: dictionary or ailab.utils.config object
                - needed keys: 
                    - no additional keys needed, just for backup
            - vocab: instance of ailab.text.vocab
            - entity_dict: instance of src/module/entity_dict
            - response_dict:  instance of src/module/response_dict

        Special usage:
            - str(): print the current status
            - len(): length of dialog

        '''

        self.cfg = cfg
        self.vocab = vocab
        self.response_dict = response_dict
        self.entity_dict = entity_dict #for response mask
        self.entity = {} #for current entity 
        self.entity_mask = numpy.ones((len(entity_dict.entity_maskdict),1), 'bool_') #for response mask
        self.utterances, self.responses, self.entities, self.masks = [], [], [], []


    def add_utterance(self, utterance, entity_ids=None):
        '''
            add utterance to status

            Input:
                - utterance: string or token_id list
                - entity_ids: entity id dictionary, usable only if utterance is a token_id list 

            Output:
                - if success, return True, otherwise return None

        '''
        if isinstance(utterance, str):
            #predeal utterance
            entities, tokens = self.vocab.seg_ins.get(utterance.lower())
            utterance_ids = self.vocab.sentence2id(tokens)
            if len(utterance_ids) < 1:
                return None
            entity_ids =  self.entity_dict(entities)
        else:
            utterance_ids = utterance
        self.utterances.append(utterance_ids)
        for e in entity_ids:
            self.entity[e] = entity_ids[e][0]
            #entity status and response mask
            if e in self.entity_dict.entity_maskdict:
                self.entity_mask[self.entity_dict.entity_maskdict[e], 0] = False #mask for response choose
        self.entities.append(copy.deepcopy(self.entity))
        return True


    def add_response(self, response):
        '''
            add response and apply function
            
            Input:
                - response: string
        '''
        self.responses.append(response)
        funcneeds = self.response_dict.func_need[response]
        for funcname in funcneeds:
            func = getattr(Behaviors, funcname)
            self.applyfunc(func)


    def applyfunc(self, func):
        '''
            apply function and put result to entities 

            Input:
                - func: function name
        '''
        entities_get = func(self.entity)
        for e in entities_get:
            e_id = self.entity_dict.name2id(e)
            self.entity[e_id] = self.entity_dict.value2id(e_id, entities_get[e])
            self.entities[-1][e_id] = copy.deepcopy(self.entity[e_id])
            #entity status and response mask
            if e_id in self.entity_dict.entity_maskdict:
                self.entity_mask[self.entity_dict.entity_maskdict[e_id], 0] = False #mask for response choose


    def getmask(self):
        '''
            mask for each turn of response

            No input needed
        '''
        mask_need = numpy.matmul(self.response_dict.masks['need'], self.entity_mask).reshape(1,-1)[0]
        mask_need = numpy.logical_not(mask_need)
        mask_notneed = numpy.matmul(self.response_dict.masks['notneed'], numpy.logical_not(self.entity_mask)).reshape(1,-1)[0]
        mask_notneed = numpy.logical_not(mask_notneed)
        self.masks.append(mask_need * mask_notneed)


    def __str__(self):
        '''
            print the current status
        '''
        txt = '='*60 + '\n'
        txt += 'entity: ' + str(self.entity) + '\n'
        txt += 'entity mask: ' + str(self.entity_mask.reshape(1, -1)[0]) + '\n'
        for i in range(len(self.utterances)):
            txt += '-'*20 + str(i) + '-'*20 + '\n'
            txt += 'utterance: ' + self.vocab.id2sentence(self.utterances[i]) + '\n'
            if i < len(self.responses):
                txt += 'response: ' + self.response_dict.response[self.responses[i]] + '\n'
            txt += 'entities: ' + ' '.join([self.entity_dict.entity_namedict.inv[e] for e in self.entities[i].keys()]) + '\n'
            txt += 'mask: ' + str(self.masks[i]) + '\n'
        return txt


    def __len__(self):
        '''
            length of dialog
        '''
        return len(self.utterances)


    @staticmethod 
    def torch(cfg, vocab, entity_dict, dialogs):
        '''
            staticmethod, convert dialogs to batch

            Input:
                - cfg: dictionary or ailab.utils.config object
                    - needed keys:
                        - model:
                            - max_seq_len: maximum sequence length
                            - max_entity_types: number of entity types
                            - use_gpu: if use gpu or not
                - vocab: ailab.text.vocab instance
                - entity_dict: src/module/entity_dict instance
                - dialogs: list of src/module/dialog_status instance

            Output:
                - dictionary of pytorch variables with the keys of :
                    - utterance: 2d long tensor
                    - response: 1d long tensor
                    - response_prev: 2d float tensor
                    - mask: 2d float tensor
                    - entity: 2d float tensor
                    - batch_sizes: used in lstm pack_padded_sequence to speed up
        '''
        dialog_lengths = numpy.array([len(d.utterances) for d in dialogs], 'int')
        perm_idx = dialog_lengths.argsort()[::-1]
        N_dialogs = len(dialogs)
        dialog_lengths = dialog_lengths[perm_idx]
        max_dialog_len = int(dialog_lengths[0])
        
        utterance = numpy.zeros((N_dialogs, max_dialog_len, cfg.model.max_seq_len), 'int')
        response = numpy.zeros((N_dialogs, max_dialog_len), 'int')
        entity = numpy.zeros((N_dialogs, max_dialog_len, cfg.model.max_entity_types), 'float')
        mask = numpy.zeros((N_dialogs, max_dialog_len, len(dialogs[0].masks[0])), 'float')
        response_prev = numpy.zeros((N_dialogs, max_dialog_len, len(dialogs[0].response_dict)), 'float') #previous response, onehot


        for j, idx in enumerate(perm_idx):
            dialog = dialogs[idx]
            response[j, :len(dialog.responses)] = torch.LongTensor(dialog.responses)
            for i in range(dialog_lengths[j]):
                entities = entity_dict.name2onehot(dialog.entities[i].keys()) #all entity names
                seqlen = min(cfg.model.max_seq_len, len(dialog.utterances[i]))
                utterance[j, i, :seqlen] = numpy.array(dialog.utterances[i])[:seqlen]
                entity[j, i, :] = entities
                mask[j, i, :] = dialog.masks[i].astype('float')
                if i > 0:
                    response_prev[j, i, response[j, i-1]] = 1
        
        #to variable
        utterance = torch.LongTensor(utterance)
        response = torch.LongTensor(response)
        entity = torch.FloatTensor(entity)
        mask = torch.FloatTensor(mask)
        response_prev = torch.FloatTensor(response_prev)
        if cfg.model.use_gpu:
            utterance = utterance.cuda(cfg.model.use_gpu-1)
            response = response.cuda(cfg.model.use_gpu-1)
            entity = entity.cuda(cfg.model.use_gpu-1)
            mask = mask.cuda(cfg.model.use_gpu-1)
            response_prev = response_prev.cuda(cfg.model.use_gpu-1)
        utterance = Variable(utterance)
        response = Variable(response)
        entity = Variable(entity)
        mask = Variable(mask)
        response_prev = Variable(response_prev)


        #pack them up for different dialogs
        utterance = pack_padded_sequence(utterance, dialog_lengths, batch_first=True)
        response = pack_padded_sequence(response, dialog_lengths, batch_first=True).data
        entity = pack_padded_sequence(entity, dialog_lengths, batch_first=True).data
        mask = pack_padded_sequence(mask, dialog_lengths, batch_first=True).data
        response_prev = pack_padded_sequence(response_prev, dialog_lengths, batch_first=True).data
        batch_sizes = utterance.batch_sizes
        utterance = utterance.data

        dialog_torch = {'utterance': utterance, 'response':response, 'response_prev':response_prev, 'mask':mask, 'entity':entity, 'batch_sizes':batch_sizes}
        return dialog_torch



