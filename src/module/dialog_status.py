#!/usr/bin/env python
import numpy, torch, copy
from torch.autograd import Variable
from torch import functional as F
from ..hook.behaviors import Behaviors
from ailab.utils import flat_list

class Dialog_Status:
    def __init__(self, cfg, vocab, entity_dict, response_dict):
        self.cfg = cfg
        self.vocab = vocab
        self.response_dict = response_dict
        self.entity_dict = entity_dict #for response mask
        self.entity = {} #for current entity 
        self.entity_mask = numpy.ones((len(entity_dict.entity_maskdict),1), 'bool_') #for response mask
        self.utterances, self.responses, self.entities, self.masks = [], [], [], []


    # add utterance to status
    def add_utterance(self, utterance, entities):
        self.utterances.append(utterance)
        for e in entities:
            self.entity[e] = entities[e][0]
            #entity status and response mask
            if e in self.entity_dict.entity_maskdict:
                self.entity_mask[self.entity_dict.entity_maskdict[e], 0] = False #mask for response choose
        self.entities.append(copy.deepcopy(self.entity))


    #add response and apply function
    def add_response(self, response):
        self.responses.append(response)
        funcneeds = self.response_dict.func_need[response]
        for funcname in funcneeds:
            func = getattr(Behaviors, funcname)
            self.applyfunc(func)


    #apply function and put result to entities 
    def applyfunc(self, func):
        entities_get = func(self.entity)
        for e in entities_get:
            e_id = self.entity_dict.name2id(e)
            self.entity[e_id] = self.entity_dict.value2id(e_id, entities_get[e])
            self.entities[-1][e_id] = copy.deepcopy(self.entity[e_id])
            #entity status and response mask
            if e_id in self.entity_dict.entity_maskdict:
                self.entity_mask[self.entity_dict.entity_maskdict[e_id], 0] = False #mask for response choose


    #mask for each turn of response
    def getmask(self):
        mask_need = numpy.matmul(self.response_dict.masks['need'], self.entity_mask).reshape(1,-1)[0]
        mask_need = numpy.logical_not(mask_need)
        mask_notneed = numpy.matmul(self.response_dict.masks['notneed'], numpy.logical_not(self.entity_mask)).reshape(1,-1)[0]
        mask_notneed = numpy.logical_not(mask_notneed)
        self.masks.append(mask_need * mask_notneed)


    def __str__(self):
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


    #convert an utterance to tracker input
    def __call__(self, utterance):
        entities, tokens = self.vocab.seg_ins.get(utterance.lower())
        token_ids = self.vocab.sentence2id(tokens)
        if len(token_ids) < 1:
            return None
        entity_ids =  self.entity_dict(entities)
        self.add_utterance(token_ids, entity_ids)
        self.getmask()

        utterance = numpy.ones((1, self.cfg.model.max_seq_len), 'int')*self.vocab._id_PAD
        entity = numpy.zeros((1, self.cfg.model.max_entity_types), 'float')
        mask = numpy.zeros((1, len(self.masks[0])), 'float') 
        seqlen = min(self.cfg.model.max_seq_len, len(token_ids))
        utterance[0, :seqlen] = numpy.array(token_ids)[:seqlen]
        entity[0, :] = numpy.array(self.entity_dict.name2onehot(entity_ids))
        mask[0,:] = self.masks[-1] 
        
        if self.cfg.model.use_gpu:
            utterance = Variable(torch.LongTensor(utterance).cuda(cfg.model.use_gpu-1))
            entity = Variable(torch.FloatTensor(entity).cuda(cfg.model.use_gpu-1))
            mask = Variable(torch.FloatTensor(mask).cuda(cfg.model.use_gpu-1))
        else:
            utterance = Variable(torch.LongTensor(utterance))
            entity = Variable(torch.FloatTensor(entity))
            mask = Variable(torch.FloatTensor(mask))
        
        return {'utterance': utterance, 'entity':entity, 'mask': mask}
         


    #convert to dialogs to batch
    @staticmethod 
    def torch(cfg, vocab, entity_dict, dialog):
        dialoglen = len(dialog.utterances)
        response = numpy.array(dialog.responses, 'int')
        response_prev = numpy.zeros((dialoglen, len(dialog.response_dict)), 'float') #previous response, onehot
        utterance = numpy.ones((dialoglen, cfg.model.max_seq_len), 'int')*vocab._id_PAD
        entity = numpy.zeros((dialoglen, cfg.model.max_entity_types), 'float')
        mask = numpy.zeros((dialoglen, len(dialog.masks[0])), 'float') 

        for i in range(dialoglen):
            entities = entity_dict.name2onehot(dialog.entities[i].keys()) #all entity names
            seqlen = min(cfg.model.max_seq_len, len(dialog.utterances[i]))
            utterance[i, :seqlen] = numpy.array(dialog.utterances[i])[:seqlen]
            entity[i, :] = numpy.array(entities)
            mask[i, :] = dialog.masks[i]
            if i > 0:
                response_prev[i, response[i-1]] = 1

        if cfg.model.use_gpu:
            utterance = Variable(torch.LongTensor(utterance).cuda(cfg.model.use_gpu-1))
            response = Variable(torch.LongTensor(response).cuda(cfg.model.use_gpu-1))
            response_prev = Variable(torch.FloatTensor(response_prev).cuda(cfg.model.use_gpu-1))
            entity = Variable(torch.FloatTensor(entity).cuda(cfg.model.use_gpu-1))
            mask = Variable(torch.FloatTensor(mask).cuda(cfg.model.use_gpu-1))
        else:
            utterance = Variable(torch.LongTensor(utterance))
            response = Variable(torch.LongTensor(response))
            response_prev = Variable(torch.FloatTensor(response_prev))
            entity = Variable(torch.FloatTensor(entity))
            mask = Variable(torch.FloatTensor(mask))
        dialog_torch = {'utterance': utterance, 'response':response, 'response_prev':response_prev, 'mask':mask, 'entity':entity}
        return dialog_torch



