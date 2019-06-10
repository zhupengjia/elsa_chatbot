#!/usr/bin/env python
import os, numpy, h5py, torch, re
from string import punctuation
from tqdm import tqdm
from torch.utils.data import Dataset
from whatlangid import WhatLangId
from ..module.dialog_status import DialogStatus

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''


class ReaderBase(Dataset):
    """
       Reader base class for goal oriented chatbot to predeal the dialogs 

       Input:
            - vocab:  instance of nlptools.text.vocab
            - tokenizer:  instance of nlptools.text.Tokenizer
            - ner: instance of nlptools.text.ner
            - topic_manager: topic manager instance, see src/module/topic_manager
            - sentiment_analyzer: sentiment analyzer instance
            - max_seq_len: int, maximum sequence length

        Special method supported:
            - len(): return total number of responses in template
            - iterator: return data in pytorch Variable used in tracker
    """

    def __init__(self, vocab, tokenizer, ner, topic_manager, sentiment_analyzer, max_seq_len=10,
                 max_entity_types=1024, **args):
        self.tokenizer = tokenizer
        self.ner = ner
        self.vocab = vocab
        self.topic_manager = topic_manager
        self.sentiment_analyzer = sentiment_analyzer
        self.max_seq_len = max_seq_len
        self.max_entity_types = max_entity_types
        self.data = []
        self.wtl = WhatLangId()

    def get_lang(self, text, default="en"):
        """
            Get language for text
        """
        if text is None:
            return default
        try:
            return self.wrl.predict_lang(text)
        except Exception as err:
            return default

    @staticmethod
    def clean_text(text):
        '''Clean text by removing unnecessary characters and altering the format of words.'''
        text = re.sub(r"([%s])+" % punctuation, r"\1", text.lower())
        text = re.sub(r"i' ?m", "i am", text)
        text = re.sub(r"he' ?s", "he is", text)
        text = re.sub(r"she' ?s", "she is", text)
        text = re.sub(r"it' ?s", "it is", text)
        text = re.sub(r"that' ?s", "that is", text)
        text = re.sub(r"there' ?s", "that is", text)
        text = re.sub(r"what' ?s", "that is", text)
        text = re.sub(r"where' ?s", "where is", text)
        text = re.sub(r"how' ?s", "how is", text)
        text = re.sub(r"let' ?s", "let us", text)
        text = re.sub(r"\' ?ll", " will", text)
        text = re.sub(r"\' ?ve", " have", text)
        text = re.sub(r"\' ?re", " are", text)
        text = re.sub(r"\' ?d", " would", text)
        text = re.sub(r"\' ?re", " are", text)
        text = re.sub(r"\' ?il", " will", text)
        text = re.sub(r"won' ?t", "will not", text)
        text = re.sub(r"can' ?t", "cannot", text)
        text = re.sub(r"n' ?t", " not", text)
        text = re.sub(r"n'", "ng", text)
        text = re.sub(r"' ?bout", "about", text)
        text = re.sub(r"' ?til", "until", text)
        text = re.sub("(@\S*|\S*&\S*|#\S*|http\S*|\S*[\(\)\[\]\*\_]\S*)", "", text)
        text = re.sub("\S{20,}", "", text)
        text = re.sub(r'(<!--.*?-->|<[^>]*>|\. ?\. ?\.)', "", text)
        text = re.sub(r"[-()\"#/@;:<>{}`+=~|]", "", text)
        text = re.sub("[^a-zA-Z0-9%s ]"%punctuation, "", text)
        text = re.sub("^[%s]*"%punctuation, "", text)
        text = " ".join([s.strip() for s in re.split("\s", text) if s.strip()])
        return text

    @staticmethod
    def add_trace(dset, arr):
        """
            add array to h5py
        """
        old_shape = dset.shape
        new_shape = list(old_shape)
        new_shape[0] = old_shape[0] + arr.shape[0]
        dset.resize(new_shape)
        dset[old_shape[0]:new_shape[0]+1,:] = arr

    def predeal(self, data, h5file):
        """
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

        """
        if os.path.exists(h5file):
            os.remove(h5file)
        h5file = h5py.File(h5file, 'w')
            
        h5datasets = {"point": h5file.create_dataset('point', (0, 2), dtype='i', chunks=(1, 2),
                                                     compression='lzf', maxshape=(None, 2))
                      }
        n_tot = 0
        for dialog_data in tqdm(data):
            # dialog simulator
            dialog = DialogStatus.new_dialog(self.vocab, self.tokenizer, self.ner, self.topic_manager,
                                              self.sentiment_analyzer, self.max_seq_len, self.max_entity_types)
            for i_p, pair in enumerate(dialog_data):
                if dialog.add_utterance(pair[0]) is None:
                    continue
                dialog.add_response(pair[1])
            # save to hdf5
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
                    h5datasets[k] = h5file.create_dataset(k, tuple(initshape), dtype=ddata[k].dtype,
                                                          chunks=tuple(chunkshape), compression='lzf',
                                                          maxshape=tuple(maxshape))
                ReaderBase.add_trace(h5datasets[k], ddata[k])
            ReaderBase.add_trace(h5datasets['point'], numpy.array([[n_tot, ddata['entity'].shape[0]]]))
            n_tot += ddata['entity'].shape[0]
        
        return h5file

    def __len__(self):
        return len(self.data['point'])

    def __getitem__(self, index):
        data = {}
        start_point = self.data['point'][index][0]
        end_point = start_point + self.data['point'][index][1]
        for k in self.data:
            if k == 'point':
                continue
            data[k] = torch.tensor(self.data[k][start_point:end_point])
        return data



