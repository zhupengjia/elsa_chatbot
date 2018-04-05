#!/usr/bin/env python
import os
from .reader_xlsx import Reader_xlsx

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

class Reader:
    '''
        Rule based reader
        Input:
            - cfg: dictionary or ailab.utils.config object
                - dialog_file: file of rule definition. Support xlsx
    '''

    def __new__(cls, cfg):
        ext = os.path.splitext(cfg.dialog_file)[-1]
        if ext in ['.xls', '.xlsx']:
            return Reader_xlsx(cfg)
        raise('{} file not supported!!'.format(ext))

