#!/usr/bin/env python
import torch, os, numpy
import torch.optim as optim
from nlptools.utils import Config, setLogger
from nlptools.text.ner import NER
from nlptools.text.tokenizer import Tokenizer_BERT
from ..module.topic_manager import Topic_Manager
from ..module.nltk_sentiment import NLTK_Sentiment
from ..module.dialog_status import Collate_Fn
from .sentence_encoder import Sentence_Encoder
from torch.utils.data import DataLoader

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

class Supervised:
    def __init__(self, reader,  batch_size=100, num_workers=1, epochs=1000, weight_decay=0, learning_rate=0.001, saved_model="model.pt", device='cpu', save_per_epoch=1, logger=None, **tracker_args):
        '''
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
        '''
        self.reader = reader
        topic_manager = self.reader.topic_manager
        topic_name = topic_manager.get_topic()
        self.skill = topic_manager.topics[topic_name]
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
    def build(cls, config, reader_cls, skill_cls, hook=None): 
        '''
            construct model from config

            Input:
                - config: configure dictionary
                - reader_cls: class for reader
                - skill_cls: class for skill 
                - hook: hook instance, please check src/hook/babi_gensays.py for example. Default is None
        '''
        logger = setLogger(**config.logger)
        ner = NER(**config.ner)
        tokenizer = Tokenizer_BERT(**config.tokenizer) 
        vocab = tokenizer.vocab
        
        #skill
        response = skill_cls(tokenizer=tokenizer, vocab=vocab, hook=hook, max_seq_len=config.reader.max_seq_len, **config.skill)
        topic_manager = Topic_Manager()
        topic_manager.register(config.skill.name, response)

        #reader
        max_entity_types = config.model.max_entity_types if 'max_entity_types' in config.model else 1024
        reader = reader_cls(vocab=vocab, tokenizer=tokenizer, ner=ner, topic_manager=topic_manager, sentiment_analyzer=NLTK_Sentiment(), max_seq_len=config.reader.max_seq_len, max_entity_types=max_entity_types, logger=logger)
        reader.read(config.reader.train_data)

        #reader
        #sentence encoder
        encoder = Sentence_Encoder(config.tokenizer.bert_model_name)

        return cls(reader=reader, logger=logger, encoder=encoder, **config.model)


    def __init_tracker(self, **args):
        '''tracker'''
        self.skill.init_model(saved_model=self.saved_model, device=str(self.device), **args)
        
        self.optimizer = optim.Adam(self.tracker.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.start_epoch = 0

        #checkpoint
        if os.path.exists(self.saved_model):
            self.optimizer.load_state_dict(self.skill.checkpoint["optimizer"])
            self.start_epoch = self.skill.checkpoint['epoch']

        self.generator = DataLoader(self.reader, batch_size=self.batch_size, collate_fn=Collate_Fn, shuffle=True, num_workers=self.num_workers)


    def train(self):
        self.skill.model.train() #set train flag
        
        for epoch in range(self.start_epoch, self.epochs):
            for it, d in enumerate(self.generator):

                d.to(self.device)
                self.skill.model.zero_grad()
                
                y_prob, loss = self.skill.get_response(d)

                self.logger.info('{} {} {}'.format(epoch, self.epochs, loss.item()))
        
                loss.backward()
                self.optimizer.step()

            #save
            if epoch > 0 and epoch%self.save_per_epoch == 0: 
                state = {
                            'state_dict': self.skill.model.state_dict(),
                            'config_model': self.skill.model.config,
                            'epoch': epoch,
                            'optimizer': self.optimizer.state_dict()
                        }
                torch.save(state, self.saved_model)


