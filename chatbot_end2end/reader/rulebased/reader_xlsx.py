#!/usr/bin/env python
import os, pandas, re

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

class Reader_xlsx:
    '''
        dialog reader for rule based, from xlsx file

        Input:
            - dialog_file: xlsx file of rule definition

    '''

    def __init__(self, dialog_file):
        self.dialog_file = dialog_file
        self._read()

    def _read(self):
        if not os.path.exists(self.dialog_file):
            raise('{} not exists!!!'.format(self.dialog_file))
        self.data = pandas.read_excel(self.dialog_file, index_col='id')
        self.data['childID'] = self.data['childID'].apply(self.__decode_childID)

   
    def __decode_childID(self, IDs):
        '''
            convert childID to list
        '''
        if isinstance(IDs, str):
            IDs2 = []
            for i in re.split('[,，]', IDs):
                if i.isdigit():
                    IDs2.append(int(i))
                else:
                    itmp = [int(x) for x in re.split('[~-]', i) if len(x.strip())>0]
                    if len(itmp) > 1:
                        IDs2 += range(itmp[0], itmp[1]+1)
                    else:
                        IDs2.append(int(itmp[0]))
            return IDs2
        elif isinstance(IDs, int):
            return [IDs]
        elif isinstance(IDs, list):
            if isinstance(IDs[0], int):
                return IDs
            elif isinstance(IDs[0], str):
                return [int(x) for x in IDs]
        return None

