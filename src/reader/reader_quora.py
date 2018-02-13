#!/usr/bin/env python
from nlptools.utils import zload, zdump, setLogger
from nlptools.text import Segment, Embedding, Vocab
import os, pandas, sys, numpy, math, torch
from torch.autograd import Variable

class Reader_Quora(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.seg = Segment(self.cfg)
        self.emb = Embedding(self.cfg)
        self.logger = setLogger(self.cfg)
        self.vocab = Vocab(self.cfg, self.seg, self.emb)
        self.vocab.addBE()
        self.predeal()

    def predeal(self):
        if os.path.exists(self.cfg['data_cache']):
            self.logger.info('loaded quora data from cache')
            self.data = zload(self.cfg['data_cache'])
        else:
            self.logger.info('read quora data via pandas')
            self.data = pandas.read_csv(self.cfg['data_path'], sep='\t', usecols=['question1', 'question2', 'is_duplicate']).dropna()
            self.logger.info('sentence2id for question1')
            self.data['question1_id'] = self.data['question1'].apply(self.vocab.sentence2id)
            self.logger.info('sentence2id for question2')
            self.data['question2_id'] = self.data['question2'].apply(self.vocab.sentence2id)
            self.logger.info('cache vectors')
            self.vocab.get_id2vec()
            self.vocab.save()
            self.emb.save()
            zdump(self.data, self.cfg['data_cache'])
        self.N_batches = math.ceil(len(self.data['question1_id'])/self.cfg["batch_size"])

    def shuffle(self):
        self.logger.info('shuffle data')
        self.data = self.data.sample(frac=1).reset_index(drop=True)

    def __iter__(self):
        for i in range(self.N_batches):
            yield self.__getitem__(i)

    def __getitem__(self, i):
        idx = i*self.cfg["batch_size"]
        data_len = min(self.cfg['batch_size'], len(self.data['question1_id'])-idx) 
        data = {'question1': numpy.ones((data_len, self.cfg['max_seq_len']), 'int')*self.vocab._id_PAD,\
                'question2': numpy.ones((data_len, self.cfg['max_seq_len']), 'int')*self.vocab._id_PAD}
        for i in range(data_len):
            q1len = min(len(self.data['question1_id'][idx+i]), self.cfg['max_seq_len'])
            q2len = min(len(self.data['question2_id'][idx+i]), self.cfg['max_seq_len'])
            data['question1'][i][:q1len] = self.data['question1_id'][idx+i][:q1len]
            data['question2'][i][:q2len] = self.data['question2_id'][idx+i][:q2len]
        data['match'] = self.data['is_duplicate'].as_matrix()[idx:idx+data_len].astype('float')
        if self.cfg.use_gpu:
            data['question1'] = Variable(torch.LongTensor(data['question1']).cuda(self.cfg.use_gpu-1))
            data['question2'] = Variable(torch.LongTensor(data['question2']).cuda(self.cfg.use_gpu-1))
            data['match'] = Variable(torch.FloatTensor(data['match']).cuda(self.cfg.use_gpu-1))
        else:
            data['question1'] = Variable(torch.LongTensor(data['question1']))
            data['question2'] = Variable(torch.LongTensor(data['question2']))
            data['match'] = Variable(torch.FloatTensor(data['match']))
        return data 

    def __len__(self):
        return self.N_batches

