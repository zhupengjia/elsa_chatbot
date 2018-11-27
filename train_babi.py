#!/usr/bin/env python
import sys, torch, os
from src.reader.reader_babi import Reader_Babi
from src.model.supervised import Supervised
from nlptools.utils import Config, setLogger


model = Supervised('config/babi.yml', Reader_Babi)
model.read('/home/pzhu/data/dialog/babi/dialog-babi-task5-full-dialogs-trn.txt')
model.train()



