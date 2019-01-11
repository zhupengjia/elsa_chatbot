#!/usr/bin/env python
import sys, os, numpy
from nlptools.text.ner import NER
from nlptools.text import Vocab, Embedding
from ..module.response_dict import Response_Dict
from ..module.dialog_status import Dialog_Status
from ..module.entity_dict import Entity_Dict

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

class Reader_Base(object):
    '''
       Reader base class to predeal the dialogs 

       Input:
            - vocab:  instance of nlptools.text.Vocab
            - ner: instance of nlptools.text.ner.NER
            - embedding: instance of nlptools.text.Embedding
            - entity_dict: instance of ..module.entity_dict.Entity_Dict
            - hook: hook instance, see ..hook.behaviors for example
            - max_seq_len: int, maximum sequence length
            - max_entity_types: int, number of entity types
            - epochs, int, epoch for iterator, default is 100
            - batch_size, int, batch size for iterator, default is 20
            - logger: logger instance
            
        Special method supported:
            - len(): return total number of responses in template
            - iterator: return data in pytorch Variable used in tracker
    '''

    def __init__(self, vocab, ner, embedding, entity_dict, hook, max_seq_len, max_entity_types, epochs=100, batch_size=20, logger = None):
        self.logger = logger
        self.emb = embedding
        self.ner = ner
        self.ner.build_keywords_index(embedding=self.emb)
        self.vocab = vocab
        self.entity_dict = entity_dict
        self.hook = hook
        self.epochs = epochs
        self.batch_size = batch_size
        self.data = {}
   

    def build_responses(self, template_file):
        '''
            build response index for response_template

            Input:
                - template_file: template file path
        '''
        self.response_dict = Response_Dict(self.ner, self.entity_dict, template_file+".cache")
        with open(template_file) as f:
            for l in f:
                l = l.strip()
                if len(l) < 1:
                    continue
                self.response_dict.add(l)
        self.response_dict.build_index()


    def __len__(self):
        '''
            return total number of responses in template
        '''
        return len(self.response_dict)


    def get_response(self, responseid, entity=None):
        '''
            return response string by id

            Input:
                - responseid: int
                - entity: entity dictionary to format the output response

            Output:
                - response, string
        '''
        response = self.response_dict.response[responseid]
        if entity is not None:
            response = response.format(**entity)
        return response
   
     
    def predeal(self, data):
        '''
            Predeal the dialog. Please use it with your read function.
            
            Input:
                - data: dialog data, format as::
                    
                    [
	                [
		            [utterance, response],
		            ...
	                ],
	                ...
                    ]
            
            Output::

                {  
	            'utterance': [utterance token_ids],  
	            'response': [response ids]  
	            'ent_utterance':[utterance entity ids]  
	            'idrange':[  
		        [dialog_startid, dialog_endid],  
		        ...  
	            ]  
                }

        '''
        ripe = {}
        for k in ['utterance', 'response', 'ent_utterance', 'ent_response', 'idrange']:
            ripe[k] = []

        for i_d, dialog in enumerate(data):
            if self.logger is not None:
                self.logger.info('predeal dialog {}/{}'.format(i_d, len(data)))
            ripe['idrange'].append([len(ripe['utterance']), len(ripe['utterance'])+len(dialog)])
            for i_p, pair in enumerate(dialog):
                for i_s, sentence in enumerate(pair):
                    entities, tokens = self.ner.get(sentence.lower())
                    token_ids = self.vocab.sentence2id(tokens)
                    if len(token_ids) < 1:
                        entities = {}
                        tokens = [self.vocab.PAD]
                        token_ids = [self.vocab._id_PAD]
                    entity_ids = self.entity_dict(entities)
                    if i_s == 0:
                        ripe['utterance'].append(token_ids)
                        ripe['ent_utterance'].append(entity_ids)
                    else:
                        response_id = self.response_dict[tokens + list(entities.keys())]
                        ripe['ent_response'].append(entity_ids)
                        if response_id is None:
                            ripe['response'].append(0)
                        else:
                            #print('='*60)
                            #print(response_id)
                            #print(' '.join(tokens))
                            #for i in range(min(3, len(response_id))):
                            #    print('-'*30)
                            #    print(self.response_dict.response[response_id[i][0]])
                            ripe['response'].append(response_id[0])
        self.vocab.reduce_vocab()
        self.vocab.save()
        self.emb.save()
        self.entity_dict.save()
        return ripe


    def new_dialog(self):
        '''
            return a new dialog status instance
        '''
        return  Dialog_Status(self.vocab, self.ner, self.entity_dict, self.response_dict, self.hook)


    def __iter__(self):
        '''
            tracker train iterator
        '''
        for epoch in range(self.epochs):
            dialogs = []
            for n in range(self.batch_size):
                sampleid = numpy.random.randint(len(self.data['idrange']))
                idrange = self.data['idrange'][sampleid]
                #roll entity gets
                dialog_status = self.new_dialog()
                for i in range(idrange[0], idrange[1]):
                    #add to dialog status
                    if dialog_status.add_utterance(self.data['utterance'][i], self.data['ent_utterance'][i]) is None:
                        continue
                    #mask
                    dialog_status.getmask()
                    dialog_status.add_response(self.data['response'][i])
                if self.logger is not None:
                    self.logger.debug(dialog_status)
                dialogs.append(dialog_status)
            yield Dialog_Status.torch(self.vocab, self.entity_dict, dialogs, self.max_seq_len, self.max_entity_types)


