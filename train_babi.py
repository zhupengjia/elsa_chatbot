#!/usr/bin/env python
import sys, torch, os
from src.reader.reader_babi import Reader_Babi
from src.hook.behaviors import Behaviors
from src.model.supervised import Supervised
from nlptools.utils import Config


cfg = Config("config/babi.yml")
hook = Behaviors()
model = Supervised.build(cfg, Reader_Babi,  hook)
model.read(cfg.model.train_data)
model.train()



