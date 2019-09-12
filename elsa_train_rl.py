#!/usr/bin/env python
import argparse, sys

parser = argparse.ArgumentParser(description='Training script for chatbot using user simulator and policy gradiant')
parser.add_argument('-c', '--config', dest='config', help='yaml configuration file')

args = parser.parse_args()

if len(sys.argv) < 2:
    parser.print_help()
    parser.exit()

from elsabot.model.policy_gradiant import PolicyGradiant
from nlptools.utils import Config
import torch

print(args.config)

cfg = Config(args.config)
model = PolicyGradiant.build(cfg)
model.train()

