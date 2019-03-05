#!/usr/bin/env python
import sys, torch, os
from chatbot_end2end.reader.reader_cornell import Reader_Cornell
from chatbot_end2end.skills.generative_response import Generative_Response
from chatbot_end2end.model.supervised import Supervised
from nlptools.utils import Config


cfg = Config("config/cornell.yml")
model = Supervised.build(cfg, Reader_Cornell, Generative_Response)
model.train()



