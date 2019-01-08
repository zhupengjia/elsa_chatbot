#!/usr/bin/env python3
import torch, sys
import torch.nn.functional as F
import torch.nn as nn
from .sentence_encoder import Sentence_Encoder
from .model_base import Model_Base

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

class Duplicate_Embedding(Model_Base):
    '''
        sentence embedding via supervised learning , one of the training data is Quora's duplicate QA
    '''
    def __init__(self, vocab, kernel_num, kernel_size, dropout):
        super().__init__(vocab)
        self.network()

    def network(self):
        self.encoder = Sentence_Encoder(self.vocab, self.kernel_num, self.kernel_size, dropout)
        self.encoder.network()
        self.fc1 = nn.Linear(self.kernel_num*2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, question1, question2):
        q1 = self.encoder(question1)
        q2 = self.encoder(question2)
        #q-q attention
        M_att = self.attention(q1, q2)
        out = self.sigmoid(self.fc1(M_att))
        return out 


