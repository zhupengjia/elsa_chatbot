#!/usr/bin/env python
import torch.nn as nn


'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

class Generative_Tracker(nn.Module):
    '''
        Generative based chatbot
    '''
    def __init__(self, encoder):
        self.encoder = encoder
    
        


