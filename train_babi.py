#!/usr/bin/env python
import sys, torch, os
from chatbot_end2end.reader.reader_babi import Reader_Babi
from chatbot_end2end.hook.behaviors import Behaviors
from chatbot_end2end.skills.goal_response import Goal_Response
from chatbot_end2end.model.supervised import Supervised
from nlptools.utils import Config


cfg = Config("config/babi.yml")
hook = Behaviors()
model = Supervised.build(cfg, Reader_Babi, Goal_Response, hook)
model.train()



