#!/usr/bin/env python3
import torch, sys
import torch.nn.functional as F
import torch.nn as nn
from .sentiment_encoder import Sentiment_Encoder

class Duplicate_Embedding(nn.Module):
    def __init__(self, cfg, vocab):
        super().__init__()
        self.cfg = cfg
        self.vocab = vocab
        self.network()

    def network(self):
        self.encoder = Sentiment_Encoder(self.cfg, self.vocab)
        self.encoder.network()
        self.fc1 = nn.Linear(self.cfg['cnn_kernel_num']*2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, question1, question2):
        q1 = self.encoder(question1)
        q2 = self.encoder(question2)
        #q-q attention
        M = torch.bmm(q1, q2.transpose(1,2))
        M_rowsum = M.sum(dim=1)
        M_colsum = M.sum(dim=2)
        M_att = torch.cat((M_rowsum, M_colsum), 1)
        out = self.sigmoid(self.fc1(M_att))
        return out 


