#!/usr/bin/env python
import sys
from src.reader import Reader_Dialog
from nlptools.utils import Config

#config = Config('config/chatter_conversations.yml')
#data = Reader_YAML(config)
#yamlfile = '/Users/pengjia.zhu/data/dialog/chatterbot-corpus/chatterbot_corpus/data/chinese/conversations.yml'
#data.read(yamlfile, True)

config = Config('config/hr.yml')
data = Reader_Dialog(config)
data.get_responses()

locdir = '/home/pzhu/data/accenture/HR/history'
data.read(locdir)



