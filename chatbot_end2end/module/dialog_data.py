#!/usr/bin/env python
import torch
from torch.nn.utils.rnn import PackedSequence

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''


class DialogData(dict):
    def __init__(self, dic):
        """
            define and organize the torch data 

            Input:
                - dic: dictionary
        """
        super().__init__(dic)

    # dir(object)
    def __dir__(self):
        """
             overide dir(object)
        """
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
            if isinstance(dic[k], (torch.Tensor, PackedSequence)):
                dic[k] = dic[k].to(device)
            elif isinstance(dic[k], dict):
                dic[k] = DialogData.dict_to_device(dic[k], device)
        return dic
    
    @staticmethod
    def dict_to_half(dic):
        for k in dic.keys():
            if isinstance(dic[k], dict):
                dic[k] = DialogData.dict_to_half(dic[k])
            elif isinstance(dic[k], torch.Tensor) and dic[k].dtype==torch.float32:
                dic[k] = dic[k].half()
            elif isinstance(dic[k], PackedSequence) and dic[k].data.dtype==torch.float32:
                dic[k] = dic[k].half()
        return dic

    def to(self, device):
        """
            tensor in data to torch device

            Input:
                - device: torch device
        """
        self = DialogData.dict_to_device(self, device)

    def half(self):
        """
            float32 to float16
        """
        self = DialogData.dict_to_half(self)



