#!/usr/bin/env python
import glob, os, yaml, sys, re
from nlptools.utils import zload, zdump
from .reader_base import Reader_Base
import codecs

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

class Reader_Dialog(Reader_Base):
    '''
        Read from dialog history (HR chatbot history), inherit from Reader_Base 

        Input:
            - cfg: dictionary or nlptools.utils.config object
                - needed keys: Please check needed keys in Reader_Base
    '''
    def __init__(self, cfg):
        Reader_Base.__init__(self, cfg)

    def _is_Chinese(self, input_text):
        """
        Check the input_text contains Chinese character or not.
        :param input_text:
        :return: True or False
        """
        rtn = False
        for c in input_text:
            if c > u'\u4e00' and c < u'\u9fff':
                rtn = True
                break
        return rtn

    def _is_three_continuous_response(self, dialog_pairs):
        """
        Decide if the continuous three responses of the robot are the same.
        :param dialog_pairs:
        :return:
        """
        rtn = False
        response_len = len(dialog_pairs)
        for idx in range(response_len - 2):
            three_responses = [dialog_pairs[idx][1], dialog_pairs[idx + 1][1], dialog_pairs[idx + 2][1]]
            unique_len = len(list(set(three_responses)))
            if unique_len == 1:
                rtn = True
                break
        return rtn

    def _read_file(self, file_path):
        # first store all file path
        all_files = []
        for dir_path, dir_names, file_names in os.walk(file_path):
            for name in file_names:
                all_files.append(os.path.join(dir_path, name))
        print('Length of training file:', len(all_files))

        # store all dialog pairs of each file
        all_file_pairs = []
        for each_file in all_files:
            # store dialog pair of current file
            each_file_pairs = []
            with codecs.open(each_file, encoding='utf-8') as dialog_data:
                # for user or robot, what they say maybe contain several sentences
                user_says = []
                robot_says = []
                first_line = True
                for each_line in dialog_data:
                    each_line = each_line.strip()

                    if first_line and not each_line.startswith('='):  # first line should contain '='
                        # exclude invalid files
                        break
                    else:
                        first_line = False

                    if len(each_line) == 0 or each_line.startswith('<') or each_line.lower().startswith(
                            'if your session'):
                        continue
                    elif each_line.startswith('='):  # startswith('=') means a new dialog
                        # some file contains several dialogs, for example, 2018-02-02/m.arsalan.iqbal
                        if len(each_file_pairs) > 1:  # less than 1 dialog should not be saved
                            # robot_says contains three continuous same answers should be excluded
                            if not self._is_three_continuous_response(each_file_pairs):
                                all_file_pairs.append(each_file_pairs)
                        each_file_pairs = []
                        user_says = []
                        robot_says = []
                    else:
                        name_content = each_line.split(':', 1)
                        if (len(name_content) == 2) and (len(name_content[0]) > 0) and (len(name_content[1]) > 0):
                            # first process last dialog pair
                            if len(user_says) > 0 and len(robot_says) > 0:
                                each_file_pairs.append([' '.join(user_says), ' '.join(robot_says)])
                                user_says = []
                                robot_says = []

                            name = name_content[0].strip()
                            content = name_content[1].strip()
                            if name.lower() == 'robot':  # robot reply
                                # robot starts saying, so user_saying = False
                                user_saying = False
                                robot_says.append(content)
                            else:
                                # user starts saying, so user_saying = True
                                user_saying = True
                                if not self._is_Chinese(content):  # user input
                                    user_says.append(content)
                        elif len(name_content) == 1:
                            if user_saying:  # user still saying
                                user_says.append(each_line)
                            else:  # robot still saying
                                robot_says.append(each_line)

            # add the last dialog
            if len(user_says) > 0 and len(robot_says) > 0:
                each_file_pairs.append([' '.join(user_says), ' '.join(robot_says)])

            if len(each_file_pairs) > 1:  # less than 1 dialog should not be saved
                # robot_says contains three continuous same answers should be excluded
                if not self._is_three_continuous_response(each_file_pairs):
                    all_file_pairs.append(each_file_pairs)

        return all_file_pairs


    def read(self, locdir):
        '''
            Input:
                - locdir: The location of saved dialog history
        '''
        cached_pkl = os.path.join(locdir, 'dialog.pkl')
        if os.path.exists(cached_pkl):
            convs = zload(cached_pkl)
        else:
            convs = self._read_file(locdir)
            convs = self.predeal(convs)
            zdump(convs, cached_pkl)
        self.responses.build_mask()
        self.data = convs



