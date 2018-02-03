#!/usr/bin/env python3
from nlptools.text.tokenizer import Segment
import re, sys

class Entity:
    def __init__(self, cfg):
        self.ner_ins = Segment(cfg)
        self.cfg = cfg
        self.custom_regex = {}
        self.replace_blacklist = list(set(list(cfg['ner_name_replace'].values()) + list(cfg['regex'].keys()))) 

    def get_regex(self, sentence, replace = False, entities=None):
        if entities is None:
            entities = {}
        replaced = sentence
        regex = dict(**self.cfg['regex'], **self.custom_regex) 
        for reg in list(regex.keys()):
            for entity in re.finditer(regex[reg], replaced):
                if reg in entities:
                    entities[reg].append(entity.group(0))
                else:
                    entities[reg] = [entity.group(0)]
            replaced = re.sub(regex[reg], '<'+reg.upper()+'>', replaced)
        if replace:
            return entities, replaced
        else:
            return entities


    def get_ner(self, sentence, replace = False, entities=None):
        if entities is None:
            entities = {}
        replace_blacklist = list(set(list(entities.keys()) + self.replace_blacklist))
        tokens = self.ner_ins.seg(sentence)
        for i, e in enumerate(tokens['entities']):
            if len(e) > 0 and e != 'O' and not tokens['tokens'][i] in replace_blacklist:
                if e in entities:
                    entities[e].append(tokens['tokens'][i])
                else:
                    entities[e] = [tokens['tokens'][i]]
        if replace:
            replaced = []
            for i, e in enumerate(tokens['entities']):
                if len(tokens['tokens'][i]) < 1:continue
                elif tokens['tokens'][i] in ['<', '>']:
                    continue
                if tokens['tokens'][i].upper() in replace_blacklist:
                    tokens['tokens'][i] = tokens['tokens'][i].upper()
                if i > 0 and i < len(tokens['tokens'])-1 and tokens['tokens'][i-1] == '<' and tokens['tokens'][i+1] == '>':
                    replaced.append('<' + tokens['tokens'][i].upper() + '>')
                elif e == 'O' or e == '' or tokens['tokens'][i] in replace_blacklist:
                        replaced.append( tokens['tokens'][i] )
                else:
                    replaced.append( '<' + e.upper() + '>' )
            return entities, replaced

        else:
            return entities

    def get(self, sentence, entities = None):
        if entities is None:
            entities = {}
        entities, replace_regex = self.get_regex(sentence, True, entities)
        entities, tokens = self.get_ner(replace_regex, True, entities)
        return entities, tokens

