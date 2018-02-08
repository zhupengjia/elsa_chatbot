#!/usr/bin/env python
import torch, sys
import torch.nn.functional as F
import torch.nn as nn
from .sentiment_encoder import Sentiment_Encoder

class Dialog_Tracker(nn.module):
    def __init__(self, cfg, vocab):
        super().__init__()
        self.cfg = cfg
        self.vocab = vocab
        self.network()

    def network(self):
        self.encoder = Sentiment_Encoder(self.cfg, self.vocab)
        self.encoder.network()

    def forward(self, utterance, entity, mask):
        utterance = self.encoder(utterance)
        pass
