#!/usr/bin/env python3
import torch, sys
import torch.nn.functional as F
import torch.nn as nn
from .sentence_encoder import Sentence_Encoder
from .model_base import Model_Base

class Self_Embedding(Model_Base):
    def __init__(self, cfg, vocab):
        super().__init__(cfg, vocab)
        self.network()

    def network(self):
        self.encoder = Sentence_Encoder(self.cfg, self.vocab)
        self.encoder.network()
        self.fc1 = nn.Linear(self.cfg['cnn_kernel_num']*2, 1)
        

    def forward(self, question):
        q = self.encoder(question)
        #self attention
        M_att = self.attention(q, q)
        
        #not finished
        return out 


