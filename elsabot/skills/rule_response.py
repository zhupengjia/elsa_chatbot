#!/usr/bin/env python
"""
    Author: Pengjia Zhu (zhupengjia@gmail.com)
    Response skill for rule-based chatbot
"""
import numpy, pandas, random, re
from sklearn.metrics.pairwise import cosine_distances
from nlptools.text.docsim import WMDSim
from nlptools.text.embedding import Embedding
from nlptools.text.tokenizer import format_sentence
from .skill_base import SkillBase


class RuleResponse(SkillBase):
    '''
        Rule based skill
    '''
    def __init__(self, skill_name, dialogflow, tokenizer, vocab, max_seq_len, **args):
        super(RuleResponse, self).__init__(skill_name)
        self.dialogflow =dialogflow
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.max_seq_len = max_seq_len

    def __getitem__(self, response):
        """
            get response id from string

            Input:
                - response: string
        """
        response_id = self.dialogflow.search(response, target="response", n_top=1)[0]
        return response_id

    def update_mask(self, current_status):
        """
            Update response masks after retrieving utterance and before getting response

            Input:
                - current_status: dictionary of status, generated from Dialog_Status module
        """
        entity_mask = numpy.ones((len(self.dialogflow.entity_maskdict),1), 'bool_')
        for e in current_status["entity"]:
            if e in self.dialogflow.entity_maskdict:
                entity_mask[self.dialogflow.entity_maskdict[e], 0] = False #mask for response choose
        #calculate response mask
        mask_need = numpy.matmul(self.dialogflow.entity_masks['need'], entity_mask).reshape(1,-1)[0]
        mask_need = numpy.logical_not(mask_need)
        mask_notneed = numpy.matmul(self.dialogflow.entity_masks['unneed'], numpy.logical_not(entity_mask)).reshape(1,-1)[0]
        mask_notneed = numpy.logical_not(mask_notneed)
        mask = mask_need * mask_notneed

        if "childid_"+self.skill_name in current_status\
           and isinstance(current_status["childid_"+self.skill_name], numpy.ndarray):
            mask = mask * current_status["childid_"+self.skill_name]

        return mask.astype("float32")

    def get_response_by_id(self, response_id, entity=None):
        """
            return response string by response id

            Input:
                - responseid: int
                - entity: entity dictionary to format the output response

            Output:
                - response, string
        """
        if entity["RESPONSE"] is not None:
            response = entity["RESPONSE"]
        else:
            response = self.dialogflow.get_response(response_id)
        if entity is not None:
            response = response.format(**entity)
        return response

    def update_response(self, response_id, current_status):
        """
            update current response to the response status.

            Input:
                - response id: int
                - current_status: dictionary of status, generated from Dialog_Status module
        """
        # for multi response ids
        if isinstance(response_id, (numpy.ndarray, list)):
            current_status['childid_' + self.skill_name] = -10000
            current_status['response_' + self.skill_name] = response_id
            questions = [self.dialogflow.get_usersays(i) for i in response_id]
            questions = [" {}: {}".format(q[1], q[0]) for q in zip(questions, range(len(questions)))]
            current_status["entity"]["RESPONSE"] = self.confuse_reply + "\n{}".format("\n".join(questions))
            current_status["entity"]["UTTERANCE_OLD"] = current_status["entity"]["UTTERANCE"]
            return current_status

        # function call
        func_need = self.dialogflow.get_action(response_id)
        if func_need is not None:
            for funcname in func_need:
                if not funcname in self.dialogflow.actions:
                    continue
                entities_get = self.dialogflow.actions[funcname](current_status["entity"])
                for e in entities_get:
                    current_status["entity"][e] = entities_get[e]
        current_status['response_' + self.skill_name] = response_id
        current_status["entity"]["RESPONSE"] = self.get_response_by_id(response_id, entity=current_status["entity"])
        current_status['childid_' + self.skill_name] = self.dialogflow.dialogs.loc[response_id, 'child_id']
        return current_status

    def init_model(self, device='cpu', prefilter=500, score_tolerate=0.01, min_score=0.7, believe_score=0.85, confuse_reply="Please select:", **args):
        """
            init similarity and predeal the dialog

            Input:
                - device: string, model location, default is 'cpu'
                - see ..model.similarity for more parameters if path of saved_model not existed
        """
        self.prefilter = prefilter
        self.score_tolerate = score_tolerate
        self.min_score = min_score
        self.believe_score = believe_score
        self.confuse_reply = confuse_reply
        self.vocab.embedding = Embedding(**args)
        self.similarity = WMDSim(vocab=self.vocab, **args)
        self.similarity.to(device)
        user_says_series = self.dialogflow.dialogs["user_say_tokens"]

        fallback_says = user_says_series.isnull()
        fallback_says = fallback_says[fallback_says].index.values
        
        # flat the usersays and get embedding
        sentence, sentence_masks, dialog_ids = [], [], []
        for i, sl in enumerate(user_says_series.dropna().tolist()):
            for s in sl:
                _tmp = format_sentence(s,
                                       vocab=self.vocab,
                                       tokenizer=self.tokenizer,
                                       max_seq_len=self.max_seq_len)
                if _tmp is None: continue
                sentence.append(_tmp[0])
                sentence_masks.append(_tmp[1])
                dialog_ids.append(user_says_series.index[i])
        self.usersays_index = {"user_emb":self.similarity(sentence, sentence_masks),
                               "ids":numpy.array(dialog_ids),
                               "fallback": fallback_says}

    def get_response(self, status_data, current_status, incre_state=None, **args):
        '''
            get response from utterance

            Input:
                - status_data: data converted from dialog status
                - current_status: dictionary of status, generated from dialog_status module
                - incre_state: incremental state, default is None
        '''
        # for special interaction
        if "childid_" + self.skill_name in current_status and current_status["childid_" + self.skill_name] == -10000:
            response_id = re.sub("\D", "", current_status["entity"]["UTTERANCE"])
            if len(response_id) < 1:
                return self.get_fallback(current_status)
            response_id = int(response_id)
            if response_id >= 0 and response_id <= len(current_status['response_' + self.skill_name]):
                response_id = current_status['response_' + self.skill_name][response_id]
                current_status['response_' + self.skill_name] = response_id
                usersay = current_status['entity']["UTTERANCE_OLD"]
                self.dialogflow.add_usersay(response_id, usersay)
                _tmp = format_sentence(usersay,
                                       vocab=self.vocab,
                                       tokenizer=self.tokenizer,
                                       max_seq_len=self.max_seq_len)
                if _tmp is not None:
                    _tmp_emb = self.similarity([_tmp[0]], [_tmp[1]])
                    self.usersays_index["user_emb"] = numpy.array(list(self.usersays_index["user_emb"]) + [_tmp_emb[0]])
                    self.usersays_index["ids"] = numpy.append(self.usersays_index["ids"], response_id)
                return response_id, 1
            return self.get_fallback(current_status)

        utterance, utterance_mask = status_data["utterance"].data, status_data["utterance_mask"].data

        # filter ids by tfidf
        if self.usersays_index["ids"].shape[0] > self.prefilter:
            utterance_ids = utterance[0].cpu().detach().numpy()[utterance_mask[0].cpu().detach().numpy().astype("bool_")]
            utterance_tokens = self.vocab.id2words(utterance_ids[1:-1])

            ids = self.dialogflow.search(utterance_tokens, target="user_says", n_top=self.prefilter)
            filter_idx = numpy.in1d(self.usersays_index["ids"], ids)
        else:
            filter_idx = True

        # search by word embedding
        utterance_embedding = self.similarity.get_embedding(utterance, utterance_mask)

        response_mask = status_data["response_mask_"+self.skill_name].data[0].cpu().detach().numpy().astype("bool_")
        response_mask = response_mask[self.usersays_index["ids"]]

        filtered_data = {"user_emb":self.usersays_index["user_emb"][response_mask*filter_idx],
                         "ids":self.usersays_index["ids"][response_mask*filter_idx]}

        if len(filtered_data["ids"]) < 1:
            return self.get_fallback(current_status)

        similarity = self.similarity.similarity(utterance_embedding, filtered_data["user_emb"])[0]

        response_ids = filtered_data["ids"][similarity >= self.min_score]
        similarity = similarity[similarity >= self.min_score]
        if len(response_ids) < 1:
            return self.get_fallback(current_status)

        # sort the answers
        id_order = numpy.argsort(similarity)[::-1]
        response_ids = response_ids[id_order]
        similarity = similarity[id_order]

        # if maximum similarity is larger than believe score, then return one result
        if similarity[0] >= self.believe_score:
            max_similarity = similarity[0]
            response_ids = response_ids[similarity >= max(max_similarity-self.score_tolerate, self.believe_score)]
            return numpy.random.choice(response_ids), max_similarity

        #if maximum similarity is less than believe score, but larger than minimum score, then ask user to choice one 
        response_ids = response_ids[similarity >= max(similarity[0]-self.score_tolerate, self.min_score)]
        similarity = similarity[similarity >= max(similarity[0]-self.score_tolerate, self.min_score)]

        response_ids = numpy.unique(response_ids[:5])

        if len(response_ids) < 2:
            return response_ids[0], similarity[0]

        return response_ids, similarity[0]


    def get_fallback(self, current_status):
        '''
            Get fallback feedback
        '''
        if len(self.usersays_index["fallback"]) < 1:
            return None, 0
        return numpy.random.choice(self.usersays_index["fallback"]), 1

