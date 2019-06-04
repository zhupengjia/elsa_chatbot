#!/usr/bin/env python
import os, h5py, re
from .reader_base import ReaderBase

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

class ReaderCornell(ReaderBase):
    '''
        Read from Cornell training data, inherit from Reader_Base

        Input:
            - see Reader_Base
    '''
    
    def __init__(self, **args):
        ReaderBase.__init__(self, **args)

    def read(self, filepath):
        cached_data = filepath + '.h5'
        if os.path.exists(cached_data) and os.path.getsize(cached_data) > 102400:
            self.data = h5py.File(cached_data, 'r')
            return

        line_path = os.path.join(filepath, "movie_lines.txt")
        conv_path = os.path.join(filepath, "movie_conversations.txt")

        # Create a dictionary to map each line's id with its text
        with open(line_path, encoding='utf-8', errors='ignore') as f:
            id2line = {}
            for line in f:
                line = line.strip()
                _line = line.split(' +++$+++ ')
                if len(_line) == 5:
                    id2line[_line[0]] = _line[4]

        max_conv_len = 10
        def convs_iter(id2line):
            with open(conv_path, encoding='utf-8', errors='ignore') as f:
                for line in f:
                    # Create conversations' lines' ids.
                    line = line.strip()
                    _line = line.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
                    conv_map = _line.split(',')
                    # Sort the sentences into questions (inputs) and answers (targets)
                    conv = []
                    for i in range(len(conv_map)-1):
                        if not conv_map[i] in id2line or not conv_map[i+1] in id2line:
                            continue
                        utterance = ReaderBase.clean_text(id2line[conv_map[i]])
                        response = ReaderBase.clean_text(id2line[conv_map[i+1]])
                        if len(conv) > max_conv_len:
                            yield conv[::2]
                            yield conv[1::2]
                            conv = []
                        conv.append([utterance, response])

                    if len(conv) > 0:
                        #use both side of dialogs
                        yield conv[::2]
                        if len(conv) > 1:
                            yield conv[1::2]

        #for c in convs_iter(id2line):
        #    print(c)
        #sys.exit()
        self.data = self.predeal(convs_iter(id2line), cached_data)

