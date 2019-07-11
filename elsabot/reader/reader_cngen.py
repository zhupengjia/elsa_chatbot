#!/usr/bin/env python
import os, h5py, re, gzip, sys
from nlptools.utils import zload
from .reader_base import ReaderBase

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

class ReaderCNGen(ReaderBase):
    """
        Read from twitter conversation ids
    """

    def __init__(self, **args):
        ReaderBase.__init__(self, **args)

    @staticmethod
    def clean_text(text):
        from zhon.hanzi import punctuation
        '''Clean text by removing unnecessary characters and altering the format of words.'''
        text = re.sub(r"([%s])+" % punctuation, r"\1", text.lower())
        text = re.sub("(@\S*|\S*&\S*|#\S*|http\S*|\S*[\(\)\[\]\*\_]\S*)", "", text)
        text = re.sub(r'(<!--.*?-->|<[^>]*>|\. ?\. ?\.)', "", text)
        text = re.sub(r"[-()\"#/@;:<>{}`+=~|]", "", text)
        text = re.sub("^[%s]*"%punctuation, "", text)
        text = " ".join([s.strip() for s in re.split("\s", text) if s.strip()])
        return text

    def read(self, filepath):
        h5file = os.path.splitext(filepath)
        cached_data = filepath + '.h5'
        if os.path.exists(cached_data) and os.path.getsize(cached_data) > 102400:
            self.data = h5py.File(cached_data, 'r', libver='latest', swmr=True)
            return

        def convs_iter():
            with gzip.open(filepath, mode="rt", encoding="utf-8") as f:
                for line in f:
                    texts = re.split("\t", line)
                    if len(texts) < 2:
                        continue
                    utterance = ReaderCNGen.clean_text(texts[0])
                    response = ReaderCNGen.clean_text(texts[1])
                    if len(utterance) < 1 or len(response) < 1:
                        continue
                    yield [[utterance, response]]

        #for c in convs_iter():
        #    print(c)
        #sys.exit()
        self.data = self.predeal(convs_iter(), cached_data)

