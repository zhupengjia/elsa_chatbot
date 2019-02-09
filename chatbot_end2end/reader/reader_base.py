#!/usr/bin/env python
import sys, os, numpy
from torch.utils.data import Dataset
from ..module.dialog_status import Dialog_Status

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

def Collate_Fn(batch):
    print(batch)
    sys.exit()

class Reader_Base(Dataset):
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

    def __init__(self, vocab, tokenizer, ner, topic_manager, sentiment_analyzer, max_seq_len=10, max_entity_types=1024, logger=None):
        self.logger = logger
        self.tokenizer = tokenizer
        self.ner = ner
        self.vocab = vocab
        self.topic_manager = topic_manager
        self.sentiment_analyzer = sentiment_analyzer
        self.max_seq_len = max_seq_len
        self.max_entity_types = max_entity_types
        self.data = []
   
     
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
        dialogs = []
        for i_d, dialog_data in enumerate(data):
            if self.logger is not None:
                self.logger.info('predeal dialog {}/{}'.format(i_d, len(data)))
            dialog = Dialog_Status.new_dialog(self.vocab, self.tokenizer, self.ner, self.topic_manager, self.sentiment_analyzer, self.max_seq_len, self.max_entity_types)
            for i_p, pair in enumerate(dialog_data):
                dialog.add_utterance(pair[0])
                dialog.add_response(pair[1])
            dialogs.append(dialog.data())
        return dialogs
   

    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        return self.data[index]



