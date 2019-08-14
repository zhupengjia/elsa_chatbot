#!/usr/bin/env python
import argparse, sys

parser = argparse.ArgumentParser(description='Training script for chatbot')
parser.add_argument('-c', '--config', dest='config', help='yaml configuration file')

args = parser.parse_args()

if len(sys.argv) < 2:
    parser.print_help()
    parser.exit()

from elsabot.model.supervised import Supervised
from nlptools.utils import Config
import torch

print(args.config)

cfg = Config(args.config)
model = Supervised.build(cfg)
model.train()


