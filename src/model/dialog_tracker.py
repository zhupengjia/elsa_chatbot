#!/usr/bin/env python
import torch, sys
import torch.nn.functional as F
import torch.nn as nn
from .sentence_encoder import Sentence_Encoder

class Dialog_Tracker(nn.Module):
    def __init__(self, cfg, vocab):
        super().__init__()
        self.cfg = cfg
        self.vocab = vocab
        self.network()

    def network(self):
        self.encoder = Sentence_Encoder(self.cfg, self.vocab)
        self.encoder.network()
        


    def forward(self, utterance, entity, mask):
        utterance = self.encoder(utterance)
        pass
