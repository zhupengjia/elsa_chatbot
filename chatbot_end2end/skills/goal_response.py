#!/usr/bin/env python
import sys, re, numpy, os, torch
from nlptools.text import VecTFIDF
from nlptools.text.ngrams import Ngrams
from nlptools.utils import flat_list
from .skill_base import SkillBase
from ..model.dialog_tracker import Dialog_Tracker

"""
    Author: Pengjia Zhu (zhupengjia@gmail.com)
"""


class GoalResponse(SkillBase):
    """
        Response skill for goal oriented chatbot. Used to index the response template, get the most closed
        response_template from a response string, call model to get the response from utterance

        First you need a response template file, the format in each line is:  
            - needed_entity | notneeded_entity | func_call | response  
                - needed_entity means this response is available only those entities existed  
                - notneeded_entity means this response is not available if those entities existed  
                - func_call is the needed function defined in hook function before return the response. 

        The class will build a tf-idf index for template, the __getitem__ method is to get the most closed response
        via the tf-idf algorithm.(only used for training, the response string in training data will convert to a response id via tfidf search)

        Input:
            - tokenizer: instance of nlptools.text.tokenizer
            - hook: hook instance, please check src/hook/babi_gensays.py for example
            - template_file: template file path

        Special usage:
            - len(): return number of responses in template
            - __getitem__ : get most closed response id for response, input is response string
    """
    def __init__(self, tokenizer, hook, template_file, **args):
        super().__init__()
        self.tokenizer = tokenizer
        self.hook = hook
        self.build_index(template_file)

    def _add(self, response):
        """
            add a response to dictionary, only used when building the dictionary

            Input:
                - response: string, usually from response_template
        """
        response = [x.strip() for x in response.split('|')]
        if len(response) < 3:
            return
        entity_need = {}
        try:
            entity_need['need'], entity_need['notneed'], func_need, response = tuple(response)
        except Exception as err:
            print("Error: Template error!!")
            print("Error sentence: " + ' | '.join(response))
            print("The format should be: needentity | notneedentity | func | response")
            sys.exit()

        response_lite = re.sub('(\{[A-Z]+\})|(\d+)','', response)
        response_ids = self.vocab.words2id(self.tokenizer(response_lite)) 
        response_ids = numpy.concatenate(list(response_ids.values()))
        if len(response_ids) < 1:
            return
        self.response.append(response)
        self.response_ids.append(response_ids)
       
        # deal with entity_need and not need in template
        entity_need = {k:[x.strip() for x in re.split(',', entity_need[k])] for k in entity_need}
        entity_need = {k:[x.upper() for x in entity_need[k] if len(x) > 0] for k in entity_need}
        for k in self.entity_need: self.entity_need[k].append(entity_need[k])
        
        func_need = [x.strip() for x in re.split(',', func_need)]
        func_need = [x.lower() for x in func_need if len(x) > 0]
        self.func_need.append(func_need)

    def build_index(self, template_file):
        """
            build search index for response template. 
            
            Input:
                - template_file: template file path
        """
        self.response, self.response_ids, self.func_need = [], [], []
        self.entity_need = {'need':[], 'notneed':[]}
         
        cached_vocab = template_file + '.vocab'
        cached_index = template_file + ".index"
        self.vocab = Ngrams(ngrams=3, cached_vocab = cached_vocab) #response vocab, only used for searching best matched response template, independent with outside vocab.  
        self.__search = VecTFIDF(self.vocab, cached_index)
       
        # add response from template
        with open(template_file) as f:
            for l in f:
                l = l.strip()
                if len(l) < 1:
                    continue
                self._add(l)
        
        self.__search.load_index(self.response_ids)
        self.vocab.save()
        self._build_mask()

    def __len__(self):
        """
            get number of responses
        """
        return len(self.response)

    def update_mask(self, current_status):
        """
            Update response masks after retrieving utterance and before getting response
            
            Input:
                - current_status: dictionary of status, generated from Dialog_Status module
        """
        entity_mask = numpy.ones((len(self.entity_maskdict),1), 'bool_')
        for e in current_status["entity"]:
            if e in self.entity_maskdict:
                entity_mask[self.entity_maskdict[e], 0] = False #mask for response choose
        #calculate response mask
        mask_need = numpy.matmul(self.masks['need'], entity_mask).reshape(1,-1)[0]
        mask_need = numpy.logical_not(mask_need)
        mask_notneed = numpy.matmul(self.masks['notneed'], numpy.logical_not(entity_mask)).reshape(1,-1)[0]
        mask_notneed = numpy.logical_not(mask_notneed)
        mask = mask_need * mask_notneed
        return mask.astype("float32") 
    
    def _build_mask(self):
        """
            build entity mask of response template, converted from the template
        """
        entity_maskdict = sorted(list(set(flat_list(flat_list(self.entity_need.values())))))
        entity_maskdict = dict(zip(entity_maskdict, range(len(entity_maskdict))))
        self.masks = {'need': numpy.zeros((len(self.response), len(entity_maskdict)), 'bool_'),
                      'notneed': numpy.zeros((len(self.response), len(entity_maskdict)), 'bool_')}
        self.entity_maskdict = entity_maskdict 
        for i in range(len(self.entity_need['need'])):
            for e in self.entity_need['need'][i]:
                self.masks['need'][i, entity_maskdict[e]] = True
        for i in range(len(self.entity_need['notneed'])):
            for e in self.entity_need['notneed'][i]:
                self.masks['notneed'][i, entity_maskdict[e]] = True

    def response2onehot(self, response_id):
        """
            convert a response id to onehot present. used in dialog_tracker

            Input:
                - response_id: int

            Output:
                - 1d numpy array
        """
        response = numpy.zeros(len(self.response), 'float')
        response[response_id] = 1
        return response

    def __getitem__(self, response):
        """
            get most closed response id from templates
            
            Input:
                - response: string

            Output:
                - response_id, int. If not found return None.
        """
        response_lite = re.sub('(\{[A-Z]+\})|(\d+)','', response)
        response_ids = self.vocab.words2id(self.tokenizer(response_lite)) 
        response_ids = numpy.concatenate(list(response_ids.values()))
        if len(response_ids) < 1:
            return 0
        result = self.__search.search_index(response_ids, topN=1)
        if len(result) > 0:
            return result[0][0]
        else:
            return 0

    def init_model(self, saved_model="dialog_tracker.pt", device='cpu', **args):
        """
            init dialog tracker

            Input:
                - saved_model: str, default is "dialog_tracker.pt"
                - device: string, model location, default is 'cpu'
                - see ..model.dialog_tracker.Dialog_Tracker for more parameters if path of saved_model not existed

        """
        if os.path.exists(saved_model):
            self.checkpoint = torch.load(saved_model, map_location=lambda storage, location: storage)
            self.model = Dialog_Tracker(**{**args, **self.checkpoint['config_model']}) 
            self.model.to(device)
            self.model.load_state_dict(self.checkpoint['state_dict'])
        else:
            self.model = Dialog_Tracker(Nresponses=len(self.response), **args)
            self.model.to(device)

    def get_response_by_id(self, response_id, entity=None):
        """
            return response string by response id

            Input:
                - responseid: int
                - entity: entity dictionary to format the output response

            Output:
                - response, string
        """
        response = self.response[response_id]
        if entity is not None:
            response = response.format(**entity)
        return response

    def get_response(self, current_status):
        """
            predict response value from current status

            Input:
                - current_status: dictionary of status, generated from Dialog_Status module
        """
        if self.model.training: 
            return self.model(current_status)
        else:
            y_prob = self.model(current_status)
            _, y_pred = torch.max(y_prob.data, 1)
            y_pred = int(y_pred.cpu().numpy()[-1])
            return y_pred

    def update_response(self, skill_name, response_id, current_status):
        """
            update current response to the response status. 
            
            Input:
                - skill_name: string, name of current skill
                - response id: int
                - current_status: dictionary of status, generated from Dialog_Status module
        """
        # function call
        for funcname in self.func_need[response_id]:
            func = getattr(self.hook, funcname)
            entities_get = func(current_status["entity"])
            for e in entities_get:
                current_status["entity"][e] = entities_get[e]

        current_status['response_' + skill_name] = response_id
        current_status["response_string"] = self.get_response_by_id(response_id, entity=current_status["entity"])
        return current_status
