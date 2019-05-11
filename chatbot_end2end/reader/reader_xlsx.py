#!/usr/bin/env python
'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)

    Reader for dialog flow excel format config
'''
import os, re, pandas, random, numpy
from nlptools.utils import decode_child_id, flat_list
from nlptools.text.ngrams import Ngrams
from nlptools.text import VecTFIDF
from nlptools.text.tokenizer import Tokenizer_Simple
from nlptools.text.ner import NER


class ReaderXLSX:
    '''
        dialog reader from xlsx file

        Input:
            - dialog_file: xlsx file of rule definition
            - tokenizer: instance iof tokenizer
            - ner_config: dictionary of ner config, used to replace the related words in sentence, default is None
    '''

    def __init__(self, dialog_file, tokenizer=None, ner_config=None):
        self.dialog_file = dialog_file
        if not os.path.exists(self.dialog_file):
            raise('{} not exists!!!'.format(self.dialog_file))
        self.tokenizer = Tokenizer_Simple() if tokenizer is None else tokenizer
        self.index = None

        self.entities = self._parse_entity()
        
        if ner_config is not None:
            self.ner = NER(**ner_config, **self.entities) 
        else:
            self.ner = None

        self.dialogs = self._parse_dialog()
        self.actions = self._parse_action()

    def __len__(self):
        """
            return number of intents
        """
        return self.dialogs.index.max() + 1

    def get_action(self, idx):
        """
            return action list via intent id

            Input:
                - id: int, intent id
        """
        return self.dialogs.loc[idx, "action"]

    def get_response(self, idx):
        """
            return response template via idx. If multiple response template available, will randomly return one

            Input:
                - id: int, intent id
        """
        if not idx in self.dialogs["response"].index:
            return None
        return random.choice(self.dialogs["response"].loc[idx])

    def _mixed_tokenizer(self, d):
        if self.ner is not None:
            _, d = self.ner.get(d, return_dict=True)
        d = re.sub('[^a-zA-Z_#$ ]', '', d)
        return self.tokenizer(d)

    def _parse_dialog(self):
        try:
            dialogs = pandas.read_excel(self.dialog_file, sheet_name="dialogs", index_col='id')
        except Exception as error:
            print(error)
            return None
        if dialogs.shape[0] < 1:
            return None
        dialogs['child_id'] = dialogs['child_id'].apply(decode_child_id)
        dialogs["needed_entity"] = dialogs["needed_entity"].apply(
            lambda x: [s.strip().upper() for s in re.split("[,| ]", x) if s.strip()]\
            if isinstance(x, str) else None)
        dialogs["unneeded_entity"] = dialogs["unneeded_entity"].apply(
            lambda x: [s.strip().upper() for s in re.split("[,| ]", x) if s.strip()]\
            if isinstance(x, str) else None)
        dialogs["user_says"] = dialogs["user_says"].apply(
            lambda x: [self._mixed_tokenizer(s.strip()) for s in re.split("\n", x) if s.strip()]\
            if isinstance(x, str) else None)
        dialogs["response"] = dialogs["response"].apply(
            lambda x: [s.strip() for s in re.split("\n", x) if s.strip()]\
            if isinstance(x, str) else None)
        dialogs["action"] = dialogs["action"].apply(
            lambda x: [s.strip() for s in re.split("[\s,|]", x) if s.strip()]
            if isinstance(x, str) else None)
        self.index = {}
        self.index["response"] = self._build_index(dialogs["response"])
        self.index["user_says"] = self._build_index(dialogs["user_says"])

        dialog_len = dialogs.index.max() + 1
        
        def _build_child_mask(cid):
            if cid is None:
                return None
            mask = numpy.zeros(dialog_len, 'bool_') 
            for d in cid:
                mask[d] = True
            return mask

        dialogs['child_id'] = dialogs['child_id'].apply(_build_child_mask)

        self.entity_maskdict, self.entity_masks = \
                self._build_entity_mask(dialogs[["needed_entity", "unneeded_entity"]])
        return dialogs

    def _build_index(self, data):
        data_flat, ids_flat = [], []
        data, ids = data.tolist(), list(data.index)
        for i, rl in enumerate(data):
            if rl is None:
                continue
            for r in rl:
                data_flat.append(r)
                ids_flat.append(ids[i])
        vocab = Ngrams(ngrams=3)
        search_index = VecTFIDF(vocab)
        if isinstance(data_flat[0], str):
            data_flat = [self._mixed_tokenizer(d) for d in data_flat]
        data_ids = [flat_list(vocab(d).values()) for d in data_flat]
        vocab.freeze()
        search_index.load_index(data_ids)
        return {"vocab":vocab, "index":search_index, "ids": ids_flat}

    def search(self, sentence, target="response", n_top=10):
        """
            Search the most closed response or user_says from dialogflows and return related ids
            Input:
                - sentence: string or token list
                - target: "response" or "user_says", default is "response"
                - n_top, int, default is 10
        """
        if not target in self.index:
            raise("target must be 'response' or 'user_says'")
        if isinstance(sentence, str):
            sentence = re.sub('[^^a-zA-Z ]', '', sentence)
            sentence = self.tokenizer(sentence)
        token_ids = flat_list(self.index[target]["vocab"](sentence).values())
        result = self.index[target]["index"].search_index(token_ids, topN=n_top)
        return numpy.array([self.index[target]["ids"][r[0]] for r in result])

    def _parse_entity(self):
        try:
            entities_table = pandas.read_excel(self.dialog_file,
                                               sheet_name="entities").iloc[:, 1:]
        except Exception as error:
            print(error)
            return None
        entities = {"keywords":None, "regex":None, "ner":None, "ner_name_replace":None}
        if entities_table.shape[0] < 3:
            return entities
        for i in range(entities_table.shape[1]):
            name = entities_table.columns[i]
            name_replace = entities_table.iloc[0, i]
            entity_type = entities_table.iloc[1, i]
            values = [str(x).strip() for x in entities_table.iloc[2:, i].dropna().tolist()\
                      if str(x).strip()]
            if entity_type is None or name is None:
                continue
            name = name.strip()
            entity_type = entity_type.strip().lower()
            if isinstance(name_replace, str) and name_replace:
                if entities["ner_name_replace"] is None:
                    entities["ner_name_replace"] = {}
                entities["ner_name_replace"][name] = name_replace
            if entity_type == "ner":
                if entities["ner"] is None: entities["ner"] = []
                entities["ner"].append(name)
            else:
                if len(values) < 1:
                    continue
                if entity_type == "keywords":
                    if entities["keywords"] is None:
                        entities["keywords"] = {}
                    entities["keywords"][name] = values
                elif entity_type == "regex":
                    if entities["regex"] is None:
                        entities["regex"] = {}
                    entities["regex"][name] = values[0]
        return entities

    def _parse_action(self):
        try:
            action_table = pandas.read_excel(self.dialog_file,
                                             sheet_name="actions",
                                             header=None).dropna()
        except Exception as error:
            print(error)
            return None
        if action_table.shape[0] < 1:
            return None
        actions = {}
        for i in range(action_table.shape[0]):
            name = action_table.iloc[i, 0].strip()
            action = ["    "+x for x in re.split("\n", action_table.iloc[i, 1])]
            action.insert(0, "def {}(entities):".format(name))
            action = "\n".join(action)
            exec(action)
            actions[name] = eval(name)
        return actions

    def _build_entity_mask(self, data):
        """
            build entity mask of response template, converted from the template
        """
        max_id = data.index.max() + 1
        needed_entity = flat_list(data.needed_entity.dropna().tolist())
        unneeded_entity = flat_list(data.unneeded_entity.dropna().tolist())
        entity_maskdict = sorted(list(set(needed_entity+unneeded_entity)))
        entity_maskdict = dict(zip(entity_maskdict, range(len(entity_maskdict))))

        masks = {'need': numpy.zeros((max_id, len(entity_maskdict)), 'bool_'),
                      'unneed': numpy.zeros((max_id, len(entity_maskdict)), 'bool_')}

        for i in range(data.shape[0]):
            if data.needed_entity.iloc[i] is not None:
                for e in data.needed_entity.iloc[i]:
                    masks["need"][data.index[i], entity_maskdict[e]] = True
            if data.unneeded_entity.iloc[i] is not None:
                for e in data.unneeded_entity.iloc[i]:
                    masks["unneed"][data.index[i], entity_maskdict[e]] = True
        
        return entity_maskdict, masks

if __name__ == "__main__":
    R = ReaderXLSX("../../data/babi/babi.xlsx", Tokenizer_Simple())
    #print(R.dialogs)
    #print(R.entities)
    #print(R.actions)
    #print(R.actions["getrestinfo"]({}))
    #print(R.search("here it is", "response", n_top=1))
    #print(R.get_action(8))
    #print(R.get_response(1))
