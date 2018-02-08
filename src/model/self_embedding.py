#!/usr/bin/env python3
import torch, sys
import torch.nn.functional as F
import torch.nn as nn
from .sentiment_encoder import Sentiment_Encoder

class Self_Embedding(nn.Module):
    def __init__(self, cfg, vocab):
        super().__init__()
        self.cfg = cfg
        self.vocab = vocab
        self.network()

    def network(self):
        self.encoder = Sentiment_Encoder(self.cfg, self.vocab)
        self.encoder.network()
        self.fc1 = nn.Linear(self.cfg['cnn_kernel_num']*2, 1)
        

    def forward(self, question):
        q = self.encoder(question)
        #self attention
        M = torch.bmm(q, q.transpose(1,2))
        M_rowsum = M.sum(dim=1)
        M_colsum = M.sum(dim=2)
        M_att = torch.cat((M_rowsum, M_colsum), 1)
        
        #not finished
        return out 


