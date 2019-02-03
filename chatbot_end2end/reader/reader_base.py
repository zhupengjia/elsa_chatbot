#!/usr/bin/env python
import sys, os, numpy
from ..module.dialog_status import Dialog_Status
from ..module.entity_dict import Entity_Dict

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

class Reader_Base(object):
    '''
       Reader base class for goal oriented chatbot to predeal the dialogs 

       Input:
            - tokenizer:  instance of nlptools.text.tokenizer.Tokenizer_BERT
            - ner: instance of nlptools.text.ner.NER
            - response_skill: response skill instance, see class in ../skills
            - max_entity_types: int, number of entity types
            - max_seq_len: int, maximum sequence length
            - epochs, int, epoch for iterator, default is 100
            - batch_size, int, batch size for iterator, default is 20
            - device: instance of torch.device, default is cpu device
            - logger: logger instance
            
        Special method supported:
            - len(): return total number of responses in template
            - iterator: return data in pytorch Variable used in tracker
    '''

    def __init__(self, tokenizer, ner, response_skill,  max_entity_types, max_seq_len=100, epochs=100, batch_size=20, device=None, logger = None):
        self.logger = logger
        self.tokenizer = tokenizer
        self.ner = ner
        self.device = device
        self.vocab = self.tokenizer.vocab
        self.response_skill = response_skill
        self.max_entity_types = max_entity_types
        self.max_seq_len = max_seq_len
        self.hook = hook
        self.epochs = epochs
        self.batch_size = batch_size
        self.data = {}
   
     
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
                    sentence = sentence.strip()
                    entities, sentence_replaced = self.ner.get(sentence, return_dict=True)
                    tokens = self.tokenizer(sentence_replaced)
                    
                    token_ids = self.vocab.words2id(tokens)
                    entity_ids = self.entity_dict(entities)
                    if i_s == 0:
                        ripe['utterance'].append(token_ids)
                        ripe['ent_utterance'].append(entity_ids)
                    else:
                        response_id = self.response_skill[tokens + list(entities.keys())]
                        ripe['ent_response'].append(entity_ids)
                        if response_id is None:
                            ripe['response'].append(0)
                        else:
                            #print('='*60)
                            #print(response_id)
                            #print(' '.join(tokens))
                            #print(self.response_skill.response[response_id[0]])
                            ripe['response'].append(response_id[0])
        self.entity_dict.save()
        return ripe


    def new_dialog(self):
        '''
            return a new dialog status instance
        '''
        return  Dialog_Status(self.vocab, self.tokenizer, self.ner, self.entity_dict, self.response_skill, self.hook, self.max_seq_len)


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
                    dialog_status.add_utterance(self.data['utterance'][i], self.data['ent_utterance'][i])
                    dialog_status.add_response(self.data['response'][i])
                if self.logger is not None:
                    self.logger.debug(dialog_status)
                dialogs.append(dialog_status)
            yield Dialog_Status.torch(self.entity_dict, dialogs, self.max_entity_types, device=self.device)


