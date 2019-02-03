#!/usr/bin/env python
import sys, os, numpy
from ..module.dialog_status import Dialog_Status
from ..module.entity_dict import Entity_Dict

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

class Reader_Base:
    '''
       Reader base class for goal oriented chatbot to predeal the dialogs 

       Input:
            - vocab:  instance of nlptools.text.vocab
            - tokenizer:  instance of nlptools.text.Tokenizer
            - ner: instance of nlptools.text.ner
            - topic_manager: topic manager instance, see src/module/topic_manager
            - sentiment_analyzer: sentiment analyzer instance
            - max_seq_len: int, maximum sequence length
            - logger: logger instance
            
        Special method supported:
            - len(): return total number of responses in template
            - iterator: return data in pytorch Variable used in tracker
    '''

    def __init__(self, vocab, tokenizer, ner, topic_manager, sentiment_analyzer, max_seq_len=10, logger=None):
        self.logger = logger
        self.tokenizer = tokenizer
        self.ner = ner
        self.vocab = vocab
        self.topic_manager = topic_manager
        self.max_seq_len = max_seq_len
        
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
                - list of dialog status

        '''
        ripe = {}
        for k in ['utterance', 'response', 'ent_utterance', 'ent_response', 'idrange']:
            ripe[k] = []
        dialogs = []
        for i_d, dialog_data in enumerate(data):
            if self.logger is not None:
                self.logger.info('predeal dialog {}/{}'.format(i_d, len(data)))
            dialog = Dialog_Status.new_dialog(vocab, tokenizer, ner, topic_manager, sentiment_analyzer, max_seq_len)
            for i_p, pair in enumerate(dialog_data):
                dialog.add_utterance(pair[0])
                dialog.add_response(pair[1])
            dialogs.append(dialog)    
        return ripe
   

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


