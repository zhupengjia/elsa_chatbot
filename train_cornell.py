#!/usr/bin/env python
import sys, torch, os
from chatbot_end2end.reader.reader_cornell import Reader_Cornell
from chatbot_end2end.model.generative_supervised import Generative_Supervised
from nlptools.utils import Config


cfg = Config("config/cornell.yml")
model = Generative_Supervised.build(cfg, Reader_Cornell)
#model.train()



