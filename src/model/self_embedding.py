#!/usr/bin/env python3
import torch, sys
import torch.nn.functional as F
import torch.nn as nn
from .sentence_encoder import Sentence_Encoder
from .model_base import Model_Base

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

class Self_Embedding(Model_Base):
    '''
        sentence embedding via autoencoder, not implemented
    '''
    def __init__(self, vocab, kernel_num, kernel_size, dropout=0.2):
        super().__init__(vocab)
        self.kernel_num = kernel_num
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.network()

    def network(self):
        self.encoder = Sentence_Encoder(self.vocab, self.kernel_num, self.kernel_size, dropout)
        self.encoder.network()
        self.fc1 = nn.Linear(self.kernel_num*2, 1)
        

    def forward(self, question):
        q = self.encoder(question)
        #self attention
        M_att = self.attention(q, q)
        
        #not finished
        return out 


