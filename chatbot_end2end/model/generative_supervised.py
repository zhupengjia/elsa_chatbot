#!/usr/bin/env python
import torch,os
from nlptools.utils import Config, setLogger
from .generative_tracker import Generative_Tracker
from ..skills.generative_response import Generative_Response
from nlptools.text.ner import NER
import torch.optim as optim
from nlptools.text.tokenizer import Tokenizer_BERT
from ..module.topic_manager import Topic_Manager
from ..module.nltk_sentiment import NLTK_Sentiment
from ..module.dialog_status import Collate_Fn
from .sentence_encoder import Sentence_Encoder
from torch.utils.data import DataLoader


'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

class Generative_Supervised:
    '''
        Generative based chatbot
    '''
    def __init__(self, reader, batch_size=100, num_workers=1, epochs=1000, weight_decay=0, learning_rate=0.001, saved_model="model.pt", device='cpu', logger=None, **tracker_args):
        self.reader = reader
        
        self.saved_model = saved_model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.epochs = epochs
        self.device = torch.device(device)
        self.logger = logger 

        self.__init_tracker(**tracker_args)
   

    @classmethod
    def build(cls, config, reader_cls):
        '''
            construct model from config

            Input:
                - config: configure dictionary
                - reader_cls: class for reader
    Boot to TWRP and make a backup
    Flash updated ROM base and Gapps (and Magisk if you had this as root method, followed by no-verity if you want to run decrypted)
    Reboot and enjoy
        '''
        logger = setLogger(**config.logger)
        ner = NER(**config.ner)
        tokenizer = Tokenizer_BERT(**config.tokenizer) 
        vocab = tokenizer.vocab

        #skill
        response = Generative_Response(tokenizer=tokenizer, vocab=vocab, max_seq_len=config.reader.max_seq_len) 
        topic_manager = Topic_Manager()
        topic_manager.register(config.model.skill_name, response)

        #reader
        reader = reader_cls(vocab=vocab, tokenizer=tokenizer, ner=ner, topic_manager=topic_manager, sentiment_analyzer=NLTK_Sentiment(), max_seq_len=config.reader.max_seq_len, logger=logger)
        reader.read(config.reader.train_data)
        
        #sentence encoder
        encoder = Sentence_Encoder(config.tokenizer.bert_model_name)
        
        return cls(reader=reader, logger=logger, encoder=encoder, **config.model)


    def __init_tracker(self, **args):
        self.tracker = Generative_Tracker(**args)
       
        
        self.tracker.to(self.device)

        #checkpoint
        if os.path.exists(self.saved_model):
            checkpoint = torch.load(self.saved_model, map_location=lambda storage, location: self.device)
            self.tracker.load_state_dict(checkpoint['model'])
        

        self.optimizer = optim.Adam(self.tracker.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.generator = DataLoader(self.reader, batch_size=self.batch_size, collate_fn=Collate_Fn, shuffle=True, num_workers=self.num_workers)

    
    def train(self):
        self.tracker.train() #set train flag
        
        for epoch in range(self.epochs):
            for it, d in enumerate(self.generator):

                d.to(self.device)
                self.tracker.zero_grad()

                self.tracker(d)
                
                break
                
                self.logger.info('{} {} {}'.format(epoch, self.epochs, loss.item()))
        
                loss.backward()
                self.optimizer.step()

                #save
                if epoch > 0 and epoch%1000 == 0: 
                    model_state_dict = self.tracker.state_dict()
                    torch.save(model_state_dict, self.saved_model)

            break
