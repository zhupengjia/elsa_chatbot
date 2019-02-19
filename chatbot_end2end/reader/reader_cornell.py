#!/usr/bin/env python
import glob, os, yaml, sys, re
from nlptools.utils import zload, zdump, flat_list
from .reader_base import Reader_Base

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

class Reader_Cornell(Reader_Base):
    '''
        Read from Cornell training data, inherit from Reader_Base

        Input:
            - see Reader_Base
    '''
    
    def __init__(self, **args):
        Reader_Base.__init__(self, **args)


    @staticmethod
    def clean_text(text):
        '''Clean text by removing unnecessary characters and altering the format of words.'''
        text = text.lower()
        text = re.sub(r"i'm", "i am", text)
        text = re.sub(r"he's", "he is", text)
        text = re.sub(r"she's", "she is", text)
        text = re.sub(r"it's", "it is", text)
        text = re.sub(r"that's", "that is", text)
        text = re.sub(r"what's", "that is", text)
        text = re.sub(r"where's", "where is", text)
        text = re.sub(r"how's", "how is", text)
        text = re.sub(r"\'ll", " will", text)
        text = re.sub(r"\'ve", " have", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"\'d", " would", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"won't", "will not", text)
        text = re.sub(r"can't", "cannot", text)
        text = re.sub(r"n't", " not", text)
        text = re.sub(r"n'", "ng", text)
        text = re.sub(r"'bout", "about", text)
        text = re.sub(r"'til", "until", text)
        text = re.sub(r"[-()\"#/@;:<>{}`+=~|]", "", text)
    #     text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
        text = " ".join(text.split())
        return text


    def read(self, filepath):
        cached_pkl = filepath + '.pkl'
        if os.path.exists(cached_pkl):
            self.data = zload(cached_pkl)
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
     
        # Create a list of all of the conversations' lines' ids.
        with open(conv_path, encoding='utf-8', errors='ignore') as f:
            convs_map = []
            for line in f:
                line = line.strip()
                _line = line.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
                convs_map.append(_line.split(','))
    
        # Sort the sentences into questions (inputs) and answers (targets)
        convs = []
        for conv_map in convs_map:
            conv = []
            for i in range(len(conv_map)-1):
                if not conv_map[i] in id2line or not conv_map[i+1] in id2line:
                    continue
                utterance = Reader_Cornell.clean_text(id2line[conv_map[i]])
                response = Reader_Cornell.clean_text(id2line[conv_map[i+1]])
                conv.append([utterance, response])
            if len(conv) > 0:
                convs.append(conv[::2])
                if len(conv) > 1:
                    convs.append(conv[1::2])

        self.data = self.predeal(convs)
        zdump(self.data, cached_pkl)

