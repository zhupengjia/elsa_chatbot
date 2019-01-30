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
            - dialog_file: file of rule definition. Support xlsx
    '''

    def __new__(cls, dialog_file, **args):
        ext = os.path.splitext(dialog_file)[-1]
        if ext in ['.xls', '.xlsx']:
            return Reader_xlsx(dialog_file)
        raise('{} file not supported!!'.format(ext))

