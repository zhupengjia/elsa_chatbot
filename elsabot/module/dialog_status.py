#!/usr/bin/env python
"""
    Author: Pengjia Zhu (zhupengjia@gmail.com)
"""
import copy, numpy, torch, time, re, nltk, ipdb
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from nlptools.text.tokenizer import format_sentence
from .entity_dict import EntityDict
from .dialog_data import DialogData


def dialog_collate(batch):
    """
        Collate function for torch.data.generator
    """
    if not batch:
        return []
    data = DialogData({})
    batch = [b for b in batch if b["utterance"].shape[0]>0]
    dialog_lengths = numpy.array([b["utterance"].shape[0]
                                  for b in batch], "int")
    
    perm_idx = dialog_lengths.argsort()[::-1].astype("int")
    dialog_lengths = dialog_lengths[perm_idx]
    max_dialog_len = int(dialog_lengths[0])

    for k in batch[0].keys():
        # padding
        for b in batch:
            padshape = [0]*(b[k].dim()*2)
            padshape[-1] = int(max_dialog_len - b[k].shape[0])
            b[k] = F.pad(b[k], padshape)
        # stack
        data[k] = torch.stack([b[k] for b in batch])
        # pack
        data[k] = pack_padded_sequence(data[k][perm_idx], dialog_lengths,
                                       batch_first=True)

    return data


class DialogStatus:
    def __init__(self, vocab, tokenizer, ner, topic_manager,
                 sentiment_analyzer, spell_check=None, max_seq_len=100,
                 max_entity_types=1024, rl_maxloop=20, rl_discount=0.95):
        """
        Maintain the dialog status in a dialog

        The dialog status will maintain:
            - utterance history
            - response history
            - entity mask history
            - entity history
            - current entity mask
            - current entities

        The class also provide the method to convert those data
        to torch variables

        To use this class, one must do the following steps:
            - add_utterance: add the current utterance to status,
                the entities will be extracted from utterance
            - update_response/get_response: updated the current
                response to status or get response from current_status

        Input:
            - vocab:  instance of nlptools.text.vocab
            - tokenizer:  instance of nlptools.text.Tokenizer
            - ner: instance of nlptools.text.ner
            - topic_manager: topic manager instance,
                see src/module/topic_manager
            - sentiment_analyzer: sentiment analyzer instance
            - spell_check: spell correction instance, default is None
            - max_seq_len: int, maximum sequence length
            - max_entity_types: int, maximum entity types

        Special usage:
            - str(): print the current status
            - len(): length of dialog

        """

        self.tokenizer = tokenizer
        self.ner = ner
        self.vocab = vocab
        self.topic_manager = topic_manager
        self.max_seq_len = max_seq_len
        self.max_entity_types = max_entity_types
        self.rl_maxloop = rl_maxloop
        self.rl_discount = rl_discount

        self.current_status = self.__init_status()
        self.history_status = []
        self.sentiment_analyzer = sentiment_analyzer
        self.spell_check = spell_check

        self.special_entities = set(self.__init_status().keys())


    def __init_status(self):
        initstatus = {"$RESPONSE":None,
                      "$RESPONSE_SCORE":0,
                      "$UTTERANCE":None,
                      "$UTTERANCE_LAST":None,
                      "$HISTORY": [],
                      "$TOPIC": self.topic_manager.skill_names[0],
                      "$TOPIC_NEXT": None,
                      "$TOPIC_LIST": self.topic_manager.skill_names,
                      "$SESSION":"default",
                      "$TIME": time.time(),
                      "$SESSION_RESET": False, #will reset session if true
                      "$REDIRECT_SESSION": None, #will redirect message to this userid if it is not None

                      "$TENSOR_ENTITIES": None,
                      "$TENSOR_UTTERANCE": None,
                      "$TENSOR_UTTERANCE_MASK": None,
                      "$TENSOR_RESPONSE": {},
                      "$TENSOR_RESPONSE_MASK":{},
                      "$CHILD_ID": {},
                      "$SENTIMENT": 0,
                      "$RESPONSE_SENTIMENT": 0
                     }
        return initstatus

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r"i' ?m", "i am", text)
        text = re.sub(r"he' ?s", "he is", text)
        text = re.sub(r"she' ?s", "she is", text)
        text = re.sub(r"it' ?s", "it is", text)
        text = re.sub(r"that' ?s", "that is", text)
        text = re.sub(r"there' ?s", "there is", text)
        text = re.sub(r"what' ?s", "what is", text)
        text = re.sub(r"where' ?s", "where is", text)
        text = re.sub(r"how' ?s", "how is", text)
        text = re.sub(r"let' ?s", "let us", text)
        text = re.sub(r"you' ?d", "you had", text)
        text = re.sub(r"i' ?d", "i had", text)
        text = re.sub(r"\' ?ll", " will", text)
        text = re.sub(r"\' ?ve", " have", text)
        text = re.sub(r"\' ?re", " are", text)
        text = re.sub(r"\' ?d", " would", text)
        text = re.sub(r"\' ?re", " are", text)
        text = re.sub(r"\' ?il", " will", text)
        text = re.sub(r"won' ?t", "will not", text)
        text = re.sub(r"didn' ?t", "did not", text)
        text = re.sub(r"wasn' ?t", "was not", text)
        text = re.sub(r"can' ?t", "cannot", text)
        text = re.sub(r"n' ?t", " not", text)
        text = re.sub(r"n'", "ng", text)
        text = re.sub(r"' ?bout", "about", text)
        text = re.sub(r"' ?til", "until", text)
        return text

    @classmethod
    def new_dialog(cls, vocab, tokenizer, ner, topic_manager,
                   sentiment_analyzer, spell_check=None, max_seq_len=100,
                   max_entity_types=1024):
        """
            create a new dialog
            data[k][tk] = data[k][tk].data
        """
        return cls(vocab, tokenizer, ner, topic_manager,
                   sentiment_analyzer, spell_check, max_seq_len, max_entity_types)

    @property
    def session(self):
        """
            get current session id
        """
        return self.current_status["$SESSION"]
    
    @property
    def topic(self):
        """
            get current topic
        """
        return self.current_status["$TOPIC"]

    @property
    def last_time(self):
        """
            Return the time of last response
        """
        return self.current_status["$TIME"]

    def add_utterance(self, utterance):
        """
            add utterance to status

            Input:
                - utterance: string

            Output:
                - if success, return True, otherwise return None

        """
        utterance_raw = utterance.strip()
        
        # spell correction
        utterance = self.spell_check(self.clean_text(utterance_raw)) if self.spell_check else utterance_raw
        
        if len(utterance) < 1:
            return None

        # get sentiment
        self.current_status["$SENTIMENT"] = self.sentiment_analyzer(utterance)

        self.current_status["$UTTERANCE"] = utterance
        
        # get entities
        if self.ner is None:
            utterance_replaced = utterance
        else:
            entities, utterance_replaced = self.ner.get(utterance, return_dict=True)
            for e in entities:
                # only keep first value
                self.current_status[e] = entities[e][0]
            self.current_status["$TENSOR_ENTITIES"] =\
                EntityDict.name2onehot(set(self.current_status.keys())-self.special_entities,
                                       self.max_entity_types).astype("float32")

        # utterance to id
        #print("utterance", utterance_raw, utterance, utterance_replaced, entities, sep=" | ")
        u_and_mask = format_sentence(utterance_replaced,
                                vocab=self.vocab,
                                tokenizer=self.tokenizer,
                                max_seq_len=self.max_seq_len)
        if u_and_mask is None:
            return None

        self.current_status["$TENSOR_UTTERANCE"], self.current_status["$TENSOR_UTTERANCE_MASK"] = u_and_mask

        # get topic
        #self.current_status["entity"]["TOPIC"] = self.topic_manager.get_topic(self.current_status)

        # response mask
        self.current_status = self.topic_manager.update_response_masks(
            self.current_status)

        return True

    def add_response(self, response):
        """
            add existed response, usually for training data

            Input:
                - response: string
        """
        response_raw = response.strip()
        response = self.spell_check(self.clean_text(response_raw)) if self.spell_check else response_raw
        if len(response) < 1:
            return None

        self.current_status["$RESPONSE_SENTIMENT"] = self.sentiment_analyzer(response)
        if self.ner is None:
            response_replaced = response
        else:
            _, response_replaced = self.ner.get(response, return_dict=True)

        #print("response", response_raw, response, response_replaced, sep=" | ")
        _tmp = self.topic_manager.add_response(response_replaced,
                                            self.current_status)
        if _tmp is None: 
            return None
        self.current_status = _tmp
        self.history_status.append(copy.deepcopy(self.current_status))

    def get_response(self, response_sentiment=0, device="cpu", **args):
        """
            get response from current status

            Input:
                - response_sentiment: wanted sentiment of response, default is 0
                - device: "cpu" or "cuda:x"
        """
        self.current_status["$RESPONSE_SENTIMENT"] = response_sentiment
        current_data = self.status2data()
        current_data.to(device)
        incre_state={}
        self.current_status = \
            self.topic_manager.get_response(current_data,
                                            self.current_status,
                                            incre_state=incre_state,
                                            **args)
        self.current_status["$TIME"] = time.time()
        self.current_status["$HISTORY"].append([self.current_status["$TOPIC"], self.current_status["$UTTERANCE"], self.current_status["$RESPONSE"]])
        self.current_status["$HISTORY"] = self.current_status["$HISTORY"][-5:]
        self.history_status.append(copy.deepcopy(self.current_status))
        return self.current_status['$RESPONSE'], self.current_status['$RESPONSE_SCORE']

    def get_fallback(self, **args):
        self.current_status = self.topic_manager.get_fallback(self.current_status)
        return self.current_status['$RESPONSE'], self.current_status['$RESPONSE_SCORE']

    def export_history(self):
        """
            dialog history export
        """
        dialogs = []
        for s in self.history_status:
            dialogs.append({"utterance": s['$UTTERANCE'],
                            "response": str(s['$RESPONSE']),
                            "time": s["$TIME"],
                            "session": str(s["$SESSION"]),
                            "topic": s["$TOPIC"]})
        return dialogs

    def __str__(self):
        """
            print the current status
        """
        txt = '='*60 + '\n'
        for k in self.current_status:
            txt += "{}: {}\n".format(k, str(self.current_status[k]))
        return txt

    def __len__(self):
        """
            length of dialog
        """
        return len(self.history_status)

    def data(self, skill_names=None, status_list=None):
        """
            return pytorch data for all messages in this dialog

            Input:
                - skill_names: list of topic name need to return,
                    default is None to return all available topics
                - status_list: list status need to predeal,
                    default is None for history status
        """
        if status_list is None:
            status_list = self.history_status
        n_status = len(status_list)
        status = {"entity": numpy.zeros((n_status, self.max_entity_types),
                                        'float32'),
                  "utterance": numpy.zeros((n_status, self.max_seq_len),
                                           'int'),
                  "utterance_mask": numpy.zeros((n_status, self.max_seq_len),
                                                'int'),
                  "sentiment": numpy.zeros((n_status, 2), 'float32')
                  }

        if skill_names is None:
            skill_names = self.topic_manager.skills.keys()

        for i, sts in enumerate(status_list):
            if sts["$TENSOR_UTTERANCE"] is None:
                continue
            status["entity"][i] = sts["$TENSOR_ENTITIES"]
            status["utterance"][i] = sts["$TENSOR_UTTERANCE"]
            status["utterance_mask"][i] = sts["$TENSOR_UTTERANCE_MASK"]
            status["sentiment"][i, 0] = sts["$SENTIMENT"]
            status["sentiment"][i, 1] = sts["$RESPONSE_SENTIMENT"]
            for tk in skill_names:
                for k in ["$TENSOR_RESPONSE_MASK", "$TENSOR_RESPONSE"]:
                    rkey = k+'_'+tk
                    if tk not in sts[k]:
                        continue
                    if rkey not in status:
                        if isinstance(sts[k][tk], numpy.ndarray):
                            status[rkey] = numpy.repeat(
                                numpy.expand_dims(
                                    numpy.zeros_like(sts[k][tk]), axis=0),
                                n_status, axis=0)
                        elif isinstance(sts[k][tk], (int, float, numpy.int64, numpy.int32, numpy.int, numpy.float, numpy.float32, numpy.float64)):
                            status[rkey] = numpy.zeros((n_status, 1),
                                                       type(sts[k][tk]))
                    if rkey in status:
                        status[rkey][i] = sts[k][tk]
        return status

    def reward(self, baseline=0):
        """
            get rewards
            Input:
                - baseline: baseline of reward
        """
        n_status = len(self.history_status)
        reward =  numpy.zeros((n_status, 1), 'float32')
        if n_status < self.rl_maxloop:
            reward_base = 1
        else:
            reward_base = 0
        for i, s in enumerate(self.history_status):
            reward[i, 0] = reward * self.rl_discount**(len(self.history_status)-i-1) - baseline
        return numpy.cumsum(status["reward"][::-1, :], axis=0)[::-1, :]

    def status2data(self, skill_names=None, status=None):
        """
            convert to pytorch data for one status

            Input:
                - skill_names: list of topic name need to return,
                    default is None to return all available topics
                - status: status dictionary generated from this class,
                    default is None for current status
        """
        if status is None:
            status = self.current_status
        data = self.data(skill_names, [status])
        for k in data:
            try:
                data[k] = torch.tensor(data[k])
            except Exception as err:
                print(k, err)
                ipdb.set_trace()
        return dialog_collate([data])

