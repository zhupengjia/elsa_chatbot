#!/usr/bin/env python
import numpy, torch
from torch.autograd import Variable
from ..hook.behaviors import Behaviors

class Dialog_Status:
    def __init__(self, entitydict):
        self.entitydict =entitydict
        self.entity = {}
        self.entity_mask = numpy.ones((len(entitydict),1), 'bool_')
        self.utterances, self.responses, self.entities, self.masks = [], [], [], []

    def add(self, utterance, response, entities, funcneeds):
        self.utterances.insert(0, utterance)
        self.responses.insert(0, response)
        for e in entities:
            self.entity[e] = entities[e][0]
        for funcname in funcneeds:
            func = getattr(Behaviors, funcname)
            self.applyfunc(func)
        self.entities.insert(0, self.entity)
        #entity status and response mask
        for e in entities:
            if e in self.entitydict:
                self.entity_mask[self.entitydict[e], 0] = False #mask for response choose


    #apply function and put result to entities 
    def applyfunc(self, func):
        entities_get = func(self.entity)
        for e in entities_get:
            self.entity[e] = entities_get[e]


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
            txt += 'dialog: ' + str(self.utterances[i]) + ' ' + str(self.responses[i]) + '\n'
            #txt += 'mask: ' + str(self.masks[i]) + '\n'
        return txt

    @staticmethod 
    def torch(dialogs, gpu=False):
        data = {}
        if gpu:
            data['utterance'] = Variable(torch.LongTensor())
        else:
            pass




