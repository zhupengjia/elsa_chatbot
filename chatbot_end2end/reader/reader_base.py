#!/usr/bin/env python
import os, numpy, h5py, sys
from torch.utils.data import Dataset
from ..module.dialog_status import Dialog_Status

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

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
   
    def add_trace(self, dset, arr):
        old_shape = dset.shape
        new_shape = list(old_shape)
        new_shape[0] = old_shape[0] + arr.shape[0]
        dset.resize(new_shape)
        dset[old_shape[0]:new_shape[0]+1,:] = arr

    def predeal(self, data, h5file):
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
                - h5file: string, path of h5file
            
            Output::
                - list of dialog status

        '''
        if os.path.exists(h5file):
            os.remove(h5file)
        with h5py.File(h5file, 'w') as h5file:
            #dialogs = h5file.create_dataset('dialog', (0, ), dtype="O", maxshape=(None,))
            h5datasets = {}
            dialog_point = h5file.create_dataset('point', (0,2), dtype='i', chunks=(1, 2), maxshape=(None, 2))
            N_tot = 0
            for i_d, dialog_data in enumerate(data):
                if self.logger is not None:
                    self.logger.info('predeal dialog {}'.format(i_d))
                dialog = Dialog_Status.new_dialog(self.vocab, self.tokenizer, self.ner, self.topic_manager, self.sentiment_analyzer, self.max_seq_len, self.max_entity_types)
                for i_p, pair in enumerate(dialog_data):
                    dialog.add_utterance(pair[0])
                    dialog.add_response(pair[1])
                ddata = dialog.data()
                for k in ddata.keys():
                    kshape = ddata[k].shape
                    if not k in h5datasets:
                        initshape = list(kshape)
                        initshape[0] = 0
                        chunkshape = list(kshape)
                        chunkshape[0] = 1
                        maxshape = list(kshape)
                        maxshape[0] = None
                        h5datasets[k] = h5file.create_dataset(k, tuple(initshape), dtype=ddata[k].dtype, chunks=tuple(chunkshape), maxshape=tuple(maxshape))
                    self.add_trace(h5datasets[k], ddata[k])
                self.add_trace(dialog_point, numpy.array([[N_tot, ddata['entity'].shape[0]]]))
                N_tot += ddata['entity'].shape[0]
        
        return h5py

    def __len__(self):
        return len(self.data['point'])


    def __getitem__(self, index):
        data = {}
        start_point = self.data['point'][index][0]
        end_point = start_point + self.data['point'][index][1]
        for k in self.data:
            if k == 'point':
                continue
            data[k] = torch.tensor(self.data[k][startpoint:endpoint])
        return data



