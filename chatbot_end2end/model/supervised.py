#!/usr/bin/env python
import torch, os
import torch.optim as optim
from nlptools.utils import setLogger
from nlptools.text.ner import NER
from nlptools.text.tokenizer import Tokenizer_BERT
from ..module.topic_manager import TopicManager
from ..module.nltk_sentiment import NLTKSentiment
from ..module.dialog_status import dialog_collate
from .. import reader as Reader, skills as Skills
from .sentence_encoder import Sentence_Encoder
from torch.utils.data import DataLoader

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''


class Supervised:
    def __init__(self, reader,  batch_size=100, num_workers=1, epochs=1000, weight_decay=0, learning_rate=0.001,
                 saved_model="model.pt", device='cpu', save_per_epoch=1, logger=None, **tracker_args):
        """
            Supervised learning for chatbot

            Input:
                - reader: reader instance, see ..module.reader.*.py
                - batch_size: int, default is 100
                - num_workers: int, default is 1
                - epochs: int, default is 1000
                - weight_decay: int, default is 0
                - learning_rate: float, default is 0.001
                - saved_model: str, default is "model.pt"
                - device: string of torch device, default is "cpu"
                - logger: logger instance ,default is None
                - see Tracker class for other more args
        """
        self.reader = reader
        topic_manager = self.reader.topic_manager
        self.skill_name = topic_manager.get_topic()
        self.skill = topic_manager.skills[self.skill_name]
        self.save_per_epoch = save_per_epoch

        self.saved_model = saved_model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.epochs = epochs
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')
        self.logger = logger

        self.__init_tracker(**tracker_args)

    @classmethod
    def build(cls, config):
        """
            construct model from config

            Input:
                - config: configure dictionary
        """
        logger = setLogger(**config.logger)
        if "ner" in config:
            ner = NER(**config.ner)
        else:
            ner = None
        tokenizer = Tokenizer_BERT(**config.tokenizer)
        vocab = tokenizer.vocab
        sentiment_analyzer = NLTKSentiment()

        # reader and skill class
        if not hasattr(Reader, config.reader.wrapper):
            raise RuntimeError("Error!! Reader {} not implemented!".format(config.reader.wrapper))
        if not hasattr(Skills, config.skill.wrapper):
            raise RuntimeError("Error!! Skill {} not implemented!".format(config.skill.wrapper))
        skill_cls = getattr(Skills, config.skill.wrapper)
        reader_cls = getattr(Reader, config.reader.wrapper)

        topic_manager = TopicManager()
        response_params = config.skill
        if "dialogflow" in config.skill:
            dialogflow = ReaderXLSX(config.skill.dialogflow,
                                    tokenizer=tokenizer,
                                   ner_config=ner_config) 
            entities = dialogflow.entities
            response_params.pop('dialogflow')
        else:
            dialogflow = None
            entities = None
        response_params.pop('wrapper')

        # skill
        response = skill_cls(tokenizer=tokenizer, vocab=vocab,
                             dialogflow=dialogflow,
                             max_seq_len=config.reader.max_seq_len,
                             skill_name=config.skill.name,
                             **response_params)

        topic_manager.register(config.skill.name, response)

        # reader
        max_entity_types = config.model.max_entity_types if 'max_entity_types' in config.model else 1024
        reader = reader_cls(vocab=vocab, tokenizer=tokenizer,
                            ner=ner, topic_manager=topic_manager,
                            sentiment_analyzer=sentiment_analyzer,
                            max_seq_len=config.reader.max_seq_len,
                            max_entity_types=max_entity_types, logger=logger)
        reader.read(config.reader.train_data)

        # sentence encoder
        encoder = Sentence_Encoder(config.tokenizer.bert_model_name)

        return cls(reader=reader, logger=logger, encoder=encoder, skill_name=config.skill.name, **config.model)

    def __init_tracker(self, **args):
        """
        tracker
        """
        self.skill.init_model(saved_model=self.saved_model, device=str(self.device), **args)

        self.optimizer = optim.Adam(self.skill.model.parameters(), lr=self.learning_rate,
                                    weight_decay=self.weight_decay)
        self.start_epoch = 0

        # checkpoint
        if os.path.exists(self.saved_model):
            self.optimizer.load_state_dict(self.skill.checkpoint["optimizer"])
            self.start_epoch = self.skill.checkpoint['epoch']

        self.generator = DataLoader(self.reader, batch_size=self.batch_size, collate_fn=dialog_collate,
                                    shuffle=True, num_workers=self.num_workers)

    def train(self):
        self.skill.model.train() # set train flag

        for epoch in range(self.start_epoch, self.epochs):
            for it, d in enumerate(self.generator):

                d.to(self.device)
                self.skill.model.zero_grad()

                y_prob, loss = self.skill.get_response(d)

                self.logger.info('{} {} {}'.format(it, epoch, loss.item()))

                loss.backward()
                self.optimizer.step()

            # save
            if epoch > 0 and epoch%self.save_per_epoch == 0:
                state = {
                            'state_dict': self.skill.model.state_dict(),
                            'config_model': self.skill.model.config,
                            'epoch': epoch,
                            'optimizer': self.optimizer.state_dict()
                        }
                torch.save(state, self.saved_model)


