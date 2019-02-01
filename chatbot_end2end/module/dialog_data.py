#!/usr/bin/env python
import torch, numpy
from torch.utils.data import Dataset

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

def collate_dialogs(batch):
    '''
        collate function for dialogs used for torch.utils.data.DataLoader
    '''
    pass


class Dialog_Data(Dataset):
    def __init__(self, N_dialogs, max_dialog_len, max_seq_len):
        '''
            Dialog data used to feed neural network
        '''
        utterance = numpy.zeros((N_dialogs, max_dialog_len, max_seq_len), 'int')
        utterance_mask = numpy.zeros((N_dialogs, max_dialog_len, max_seq_len), 'int') #utterance mask for BERT
        response = numpy.zeros((N_dialogs, max_dialog_len), 'int') 
        entity = numpy.zeros((N_dialogs, max_dialog_len, max_entity_types), 'float') 
        reward = numpy.zeros((N_dialogs, max_dialog_len), 'float') 
        
    def to(self, device):
        pass


    def __len__(self):
        pass

    def __getitem__(self):
        pass

