#!/usr/bin/env python3
import torch, sys
import torch.nn.functional as F
import torch.nn as nn
from .sentence_encoder import Sentence_Encoder
from .model_base import Model_Base

class Duplicate_Embedding(Model_Base):
    def __init__(self, cfg, vocab):
        super().__init__(cfg, vocab)
        self.network()

    def network(self):
        self.encoder = Sentence_Encoder(self.cfg, self.vocab)
        self.encoder.network()
        self.fc1 = nn.Linear(self.cfg['cnn_kernel_num']*2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, question1, question2):
        q1 = self.encoder(question1)
        q2 = self.encoder(question2)
        #q-q attention
        M_att = self.attention(q1, q2)
        out = self.sigmoid(self.fc1(M_att))
        return out 


