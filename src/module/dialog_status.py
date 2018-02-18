#!/usr/bin/env python
import numpy, torch, copy
from torch.autograd import Variable
from ..hook.behaviors import Behaviors
from nlptools.utils import flat_list

class Dialog_Status:
    def __init__(self, vocab, entitydict, response_dict):
        self.vocab = vocab
        self.response_dict = response_dict
        self.entitydict = entitydict #for response mask
        self.entity = {} #for current entity 
        self.entity_mask = numpy.ones((len(entitydict.entity_maskdict),1), 'bool_') #for response mask
        self.utterances, self.responses, self.entities, self.masks = [], [], [], []

    def add(self, utterance, response, entities, funcneeds):
        self.utterances.append(utterance)
        self.responses.append(response)
        for e in entities:
            self.entity[e] = entities[e][0]
        for funcname in funcneeds:
            func = getattr(Behaviors, funcname)
            self.applyfunc(func)
        self.entities.append(copy.deepcopy(self.entity))
        #entity status and response mask
        for e in entities:
            if e in self.entitydict.entity_maskdict:
                self.entity_mask[self.entitydict.entity_maskdict[e], 0] = False #mask for response choose


    #apply function and put result to entities 
    def applyfunc(self, func):
        entities_get = func(self.entity)
        for e in entities_get:
            e_id = self.entitydict.name2id(e)
            self.entity[e_id] = self.entitydict.value2id(e_id, entities_get[e])


    #mask for each turn of response
    def getmask(self, response_masks):
        mask = numpy.matmul(response_masks, self.entity_mask).reshape(1,-1)[0]
        self.masks.append(numpy.logical_not(mask))


    def __str__(self):
        txt = '='*60 + '\n'
        txt += 'entity: ' + str(self.entity) + '\n'
        txt += 'entity mask: ' + str(self.entity_mask.reshape(1, -1)[0]) + '\n'
        for i in range(len(self.utterances)):
            txt += '-'*20 + str(i) + '-'*20 + '\n'
            txt += 'utterance: ' + self.vocab.id2sentence(self.utterances[i]) + '\n'
            txt += 'response: ' + self.response_dict.response[self.responses[i]] + '\n'
            txt += 'entities: ' + ' '.join([self.entitydict.entity_namedict.inv[e] for e in self.entities[i].keys()]) + '\n'
            txt += 'mask: ' + str(self.masks[i]) + '\n'
        return txt

    @staticmethod 
    def torch(cfg, vocab, entitydict, dialogs, shuffle=False):
        data = {}
        totlen = sum([len(d.utterances) for d in dialogs])
        utterance = numpy.ones((totlen, cfg.model.max_seq_len), 'int')*vocab._id_PAD
        response = numpy.zeros(totlen, 'int')
        entity = numpy.ones((totlen, cfg.max_entity_types), 'int')
        mask = numpy.zeros((totlen, len(dialogs[0].masks[0])), 'float') 
        starti, endi = 0, 0
        for dialog in dialogs:
            endi += len(dialog.utterances)
            response[starti:endi] = numpy.array(dialog.responses, 'int')
            for i in range(len(dialog.utterances)):
                #entities = list(dialog.entities[i].keys()) + flat_list([entitydict.entity_value[dialog.entities[i][x]] for x in dialog.entities[i] if entitydict.entity_type[x]==0]) #all entity names and string type entity values
                entities = entitydict.name2onehot(dialog.entities[i].keys()) #all entity names
                seqlen = min(cfg.model.max_seq_len, len(dialog.utterances[i]))
                utterance[starti+i, :seqlen] = numpy.array(dialog.utterances[i])[:seqlen]
                entity[starti+i, :] = numpy.array(entities)
                mask[starti+i, :] = dialog.masks[i]
            starti = endi
        
        #shuffle
        if shuffle:
            ids = numpy.arange(totlen)
            numpy.random.shuffle(ids)
            response = response[ids]
            utterance = utterance[ids, :]
            entity = entity[ids, :]
            mask = mask[ids, :]
             
        if cfg.model.use_gpu:
            utterance = Variable(torch.LongTensor(utterance).cuda(cfg.model.use_gpu-1))
            response = Variable(torch.LongTensor(response).cuda(cfg.model.use_gpu-1))
            entity = Variable(torch.LongTensor(entity).cuda(cfg.model.use_gpu-1))
            mask = Variable(torch.FloatTensor(mask).cuda(cfg.model.use_gpu-1))
        else:
            utterance = Variable(torch.LongTensor(utterance))
            response = Variable(torch.LongTensor(response))
            entity = Variable(torch.LongTensor(entity))
            mask = Variable(torch.FloatTensor(mask))
        return {'utterance': utterance, 'response':response, 'mask':mask, 'entity':entity}




