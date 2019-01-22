#!/usr/bin/env python
import sys, torch, os
from nlptools.utils import Config
from nlptools.text import Tokenizer, Embedding, Vocab
from src.model.rule_based import Rule_Based
from src.hook.babi_gensays import Babi_GenSays


class InteractiveSession():
    def __init__(self):
        self.cfg = Config('config/rule_demo.yml')
        hook = Babi_GenSays(self.cfg.hook_keywords) 
        self.chatbot = Rule_Based.build(self.cfg, hook)
        self.client_id = 'Interact'

    def test(self, say):
        self.chatbot.reset(self.client_id)
        response = self.chatbot.get_reply('', self.client_id)
        print(response)
        print(self.chatbot.get_reply(say, self.client_id)) 
        

    def interact(self):
        self.chatbot.reset(self.client_id)
        response = self.chatbot.get_reply('', self.client_id)
        print(response)
        #interaction loop
        while True:
            # get input from user
            u = input(':: ')
            u = u.strip()
            if len(u) < 1:
                continue
        
            # check for exit command
            if u in ['exit', 'stop', 'quit', 'q']:
                break
        
            else:

                response = self.chatbot.get_reply(u, self.client_id)
                print(response)

                

if __name__ == '__main__':
    # create interactive session
    isess = InteractiveSession()
    # begin interaction
    isess.interact()
    #isess.test('May I help you')
     

