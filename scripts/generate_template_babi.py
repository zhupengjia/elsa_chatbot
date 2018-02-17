#!/usr/bin/env python
import os, sys
from nlptools.text.ner import NER
from nlptools.utils import Config, setLogger

datafile = '/home/pzhu/data/dialog/babi/dialog-babi-task5-full-dialogs-trn.txt'

cfg = Config('../config/babi.yml')


ner = NER(cfg.ner)


def rm_index(row):
    return [' '.join(row[0].split(' ')[1:])] + row[1:]

convs = []
with open(datafile) as f:
    conv = []
    utterance, response = '', ''
    for l in f:
        l = l.strip()
        l = rm_index(l.split('\t'))
        if l[0][:6] == 'resto_':
            utterance, response = '', ''
            continue
        if len(l) < 2:
            if len(utterance) > 0 and len(response) > 0 :
                conv.append([utterance, response])
            if len(conv) > 0 :
                convs.append(conv)
                conv = []
            utterance, response = '', ''
        else:
            if l[0] == '<SILENCE>' :
                if l[1][:8] != 'api_call':
                    response += '\n' + l[1]
            else:
                if len(utterance) > 0 and len(response) > 0 :
                    conv.append([utterance, response])
                utterance, response = '', ''
                utterance = l[0]
                response = l[1]
               # print(utterance, '|||', response)
    if len(utterance) > 0 and len(response) > 0 :
        conv.append([utterance, response])
    if len(conv) > 0:
        convs.append(conv)


responses = []
for conv in convs:
    print('='*60)
    for l in conv:
        entities, tokens = ner.get(l[1])
        responses.append(' '.join(tokens))
        print('-'*30)
        print(l[0],'\n', l[1])
responses = sorted(list(set(responses)))
with open('template.txt', 'w') as f:
    for i, r in enumerate(responses):
        f.write(r+'\n')
        print(i, r)




