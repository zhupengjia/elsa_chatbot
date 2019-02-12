#!/usr/bin/env python
import torch

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

class Dialog_Data(dict):
    def __init__(self, dic):
        '''
            define and organize the torch data 

            Input:
                - dic: dictionary
        '''
        super().__init__(dic)

    
    # dir(object)
    def __dir__(self):
        '''
             overide dir(object)
        '''
        return tuple(self)


    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError("config has no attribute '{}'".format(key))


    def __setattr__(self, key, item):
        self[key] = item


    def __delattr__(self, key):
        del self[key]

    
    @staticmethod
    def dict_to_device(dic, device):
        for k in dic.keys():
            if isinstance(dic[k], torch.Tensor):
                dic[k] = dic[k].to(device)
            elif isinstance(dic[k], dict):
                dic[k] = Dialog_Data.dict_to_device(dic[k], device)
        return dic


    def to(self, device):
        '''
            tensor in data to torch device

            Input:
                - device: torch device
        '''
        self = Dialog_Data.dict_to_device(self, device)

