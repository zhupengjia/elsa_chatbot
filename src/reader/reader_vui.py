#!/usr/bin/env python
import glob, os, json, sys, re
from nlptools.utils import setLogger
from .reader_base import Reader_Base

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

class Reader_VUI(Reader_Base):
    '''
        Read from VUI json files (compatible with old api.ai format), inherit from Reader_Base 

        Input:
            - cfg: dictionary or nlptools.utils.config object
                - needed keys: Please check needed keys in Reader_Base
    '''
    def __init__(self, cfg):
        Reader_Base.__init__(self, cfg)
        self.data = []

    def _jsonparse(self, data):
        '''
            Parse the json data

            Input:
                - data: list of json format string
        '''
        intents, intent_names, texts, entities, metas, responses  = [], [], [], [], [], []
        handlers = {}
        for d in data:
            intent = d['id']
            intent_name = d['name']
            if 'handlers' in d and d['handlers'] is not None:
                for h in d['handlers']:
                    if not h['name'] in handlers:
                        handlers[h['name']] = {}
                    if not h['value'] in handlers[h['name']]:
                        handlers[h['name']][h['value']] = []
                    handlers[h['name']][h['value']].append(intent)
            response = [x['speech'] for x in d['response']['messages']]
            if len(d['userSays']) < 1:
                intents.append(intent)
                intent_names.append(intent_name)
                texts.append('<SILENCE>')
                entities.append([])
                metas.append([])
                responses.append(response)
            else:
                for usersays in d['userSays']:
                    text, meta_alias, entity = '', [], {}
                    for s in usersays['data']:
                        if 'meta' in s and s['meta'] is not None:
                            m = re.split('@', s['meta'])[-1]
                            if m in entity:
                                entity[m].append(s['text'])
                            else:
                                entity[m] = [s['text']]
                            meta_alias.append((m, s['alias']))
                            text += ' ' + m + ' '
                        else:
                            text += s['text']
                    entity, text_replaced = self.ner.get(text, replace=True, entities = entity)
                    entities.append(entity)
                    metas.append(list(set(meta_alias)))
                    texts.append(text_replaced)
                    intents.append(intent)
                    intent_names.append(intent_name)
                    responses.append(response)
        return {'intents':intents, 'intent_names':intent_names,  'texts': texts, 'entities': entities, 'metas': metas, 'handlers':handlers, 'responses':responses}

    def read(self, jsondir):
        '''
            Input:
                - jsondir: dictionary of json files
        '''
        data = []
        for fn in glob.glob(os.path.join(jsondir, '*.json')):
            with open(fn, 'r') as f:
                data.append(json.load(f))
            print(self._jsonparse(data))
            sys.exit()




