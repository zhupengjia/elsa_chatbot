#!/usr/bin/env python
import sys, torch, os
from chatbot_end2end.reader.reader_babi import Reader_Babi
from chatbot_end2end.hook.behaviors import Behaviors
from chatbot_end2end.model.goal_supervised import Goal_Supervised
from nlptools.utils import Config


cfg = Config("config/babi.yml")
hook = Behaviors()
model = Goal_Supervised.build(cfg, Reader_Babi,  hook)
#model.train()



