#!/usr/bin/env python
import os, h5py, re, sys
from nlptools.utils import zload
from .reader_base import ReaderBase

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

class ReaderTwitter(ReaderBase):
    """
        Read from twitter conversation ids
    """

    def __init__(self, **args):
        ReaderBase.__init__(self, **args)

    def read(self, filepath):
        cached_data = filepath + '.h5'
        if os.path.exists(cached_data) and os.path.getsize(cached_data) > 102400:
            self.data = h5py.File(cached_data, 'r')
            return

        id2text = zload(os.path.join(filepath, "id2text.pkl"))

        max_conv_len = 10
        def convs_iter():
            with open(os.path.join(filepath, "twitter_ids.txt")) as f:
                for line in f:
                    ids = [int(x) for x in re.split("\s", line) if x.strip()]
                    texts = [ReaderBase.clean_text(id2text[i]) if i in id2text else None for i in ids]
                    lang = [self.get_lang(t) for t in texts]
                    if len(texts) < 1:
                        continue

                    conv, conv2 = [], []
                    for i in range(0, len(texts)):
                        if i%2 == 0:
                            if i+1 < len(texts):
                                if texts[i] and texts[i+1] and lang[i]=="en" and lang[i+1]=="en":
                                    if len(conv) > max_conv_len:
                                        yield conv
                                        conv = []
                                    conv.append([texts[i], texts[i+1]])
                                else:
                                    if len(conv) > 0:
                                        yield conv
                                    conv = []
                        else:
                            if i+1 < len(texts):
                                if texts[i] and texts[i+1] and lang[i]=="en" and lang[i+1]=="en":
                                    if len(conv2) > max_conv_len:
                                        yield conv2
                                        conv2 = []
                                    conv2.append([texts[i], texts[i+1]])
                                else:
                                    if len(conv2) > 0:
                                        yield conv2
                                        conv2 = []
                    if len(conv) > 0:
                        yield conv
                    if len(conv2) > 0:
                        yield conv2

        #for c in convs_iter():
        #    print(c)
        #sys.exit()
        self.data = self.predeal(convs_iter(), cached_data)

