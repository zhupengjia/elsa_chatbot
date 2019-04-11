#!/usr/bin/env python
import torch
from nlptools.text.tokenizer import Tokenizer_BERT
from nlptools.utils import setLogger
from nlptools.text.ner import NER
from .nltk_sentiment import NLTKSentiment
from .dialog_status import DialogStatus
from .topic_manager import TopicManager
from ..model.sentence_encoder import Sentence_Encoder
from .. import skills as Skills, hooks as Hooks


'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''


class InteractSession:
    def __init__(self, vocab, tokenizer, ner, topic_manager,
                 sentiment_analyzer, max_seq_len=100,
                 max_entity_types=1024, device='cpu', logger=None, **args):
        """
            General interact session

            Input:
                - vocab:  instance of nlptools.text.vocab
                - tokenizer:  instance of nlptools.text.Tokenizer
                - ner: instance of nlptools.text.ner
                - topic_manager: instance of topic manager,
                    see ..module.topic_manager
                - sentiment_analyzer: sentiment analyzer instance
                - max_seq_len: int, maximum sequence length
                - max_entity_types: int, maximum entity types
                - device: string of torch device, default is "cpu"
                - logger: logger instance ,default is None
        """

        super().__init__()
        self.logger = logger
        self.device = torch.device(device)
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.ner = ner
        self.sentiment_analyzer = sentiment_analyzer
        self.topic_manager = topic_manager
        self.max_seq_len = max_seq_len
        self.max_entity_types = max_entity_types
        self.dialog_status = {}

    @classmethod
    def build(cls, config):
        """
            construct session from config

            Input:
                - config: configure dictionary
        """
        # logger, tokenizer and ner
        logger = setLogger(**config.logger)
        tokenizer = Tokenizer_BERT(**config.tokenizer)
        ner = NER(**config.ner)
        vocab = tokenizer.vocab
        sentiment_analyzer = NLTKSentiment()

        # encoder
        encoder = Sentence_Encoder(config.tokenizer.bert_model_name)

        # skills
        topic_manager = TopicManager()
        for skill_name in config.skills:
            response_params = config.skills[skill_name]
            if not hasattr(Skills, config.skills[skill_name].wrapper):
                raise RuntimeError(
                    "Error!! Skill {} not implemented!".format(
                        config.skills[skill_name].wrapper))
            skill_cls = getattr(Skills, config.skills[skill_name].wrapper)
            if "hook" in config.skills[skill_name]:
                if not hasattr(Hooks, config.skills[skill_name].hook.wrapper):
                    raise RuntimeError(
                        "Error!! Hook {} not implemented!".format(
                            config.skills[skill_name].hook.wrapper))
                hook_cls = getattr(
                    Hooks,
                    config.skills[skill_name].hook.wrapper)
                if "parameters" not in config.skills[skill_name].hook:
                    hook = hook_cls()
                else:
                    hook = hook_cls(
                        **config.skills[skill_name].hook.parameters)
                response_params.pop('hook')
            else:
                hook = None
            response = skill_cls(tokenizer=tokenizer, vocab=vocab,
                                 hook=hook,
                                 max_seq_len=config.model.max_seq_len,
                                 **response_params)
            response.init_model(
                saved_model=config.skills[skill_name].saved_model,
                device=config.model.device,
                BOS_ID=vocab.BOS_ID,
                EOS_ID=vocab.EOS_ID,
                max_seq_len = self.max_seq_len,
                skill_name=skill_name, encoder=encoder)
            response.model.eval() # set to eval mode
            topic_manager.register(skill_name, response)

        return cls(vocab=vocab, tokenizer=tokenizer, ner=ner,
                   topic_manager=topic_manager, logger=logger,
                   sentiment_analyzer=sentiment_analyzer, **config.model)

    def new_dialog(self):
        return DialogStatus.new_dialog(self.vocab, self.tokenizer,
                                       self.ner, self.topic_manager,
                                       self.sentiment_analyzer,
                                       self.max_seq_len, self.max_entity_types)

    def response(self, query, session_id="default"):
        if session_id not in self.dialog_status:
            self.dialog_status[session_id] = self.new_dialog()

        if query in ["clear", "reset", "restart", "exit", "stop", "quit", "q"]:
            self.dialog_status[session_id] = self.new_dialog()
            return "reset"

        if len(query) < 1:
            return ''

        self.dialog_status[session_id].add_utterance(query)
        return self.dialog_status[session_id].get_response(self.device)

