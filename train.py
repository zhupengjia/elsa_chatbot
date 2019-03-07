#!/usr/bin/env python
import sys

def usage():
    sys.exit("Usage: ./train config_file_path")
    
if len(sys.argv) < 2:
    usage()

from chatbot_end2end.model.supervised import Supervised
from nlptools.utils import Config
import torch

cfg = Config(sys.argv[1])
model = Supervised.build(cfg)
model.train()


