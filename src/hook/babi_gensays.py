#!/usr/bin/env python
import random

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

class Babi_GenSays:
    '''
        Generate user says for babi

        Input:
            - cfg: dictionary or ailab.utils.config object
                - needed keys:
                    - ner:
                        - keywords: dictionary of {entityname: keywords list file path}, entity recognition via keywords list
    '''
    def __init__(self, cfg):
        self.cfg = cfg
        self.greeting = ['Hello', 'Hi', 'Hey', 'bye', 'Oh', 'Good morning', 'Good afternoon', 'Good evening', 'yay', 'sweetie', 'sir', 'miss', 'hola', 'ahhh', 'Oh', 'yea',  'hallo', 'awww','heh', 'hahah', 'Haha']
        self.requesting = ["I'd like to", "I love", "I am looking for", "Can I", "Should I", "Do you have", "Could you please", "", "Would you mind", "May I have", "I prefer", 'It will be']
        self.polite = ["please", "thanks", ""]
        self.polite2 = ["thanks", "thank you", "you rock", "no thanks"]
        self.entities = {}
        for entity in self.cfg.ner.keywords:
            self.entities[entity] = []
            with open(self.cfg.ner.keywords[entity]) as f:
                for l in f:
                    l = l.strip()
                    if len(l) > 0:
                        self.entities[entity].append(l)

    def __bundle(self, *replylist):
        reply = [random.choice(x) for x in replylist]
        reply = [r.strip() for r in reply]
        reply = [r for r in reply if len(r) > 0]
        return ' '.join(reply)

    def __getpeople(self):
        suffix = ['', 'people', 'person']
        return self.__bundle(self.entities['PARTY_SIZE'], suffix)
    
    def __getcuisine(self):
        prefix = ['', 'with', 'take']
        suffix = ['', 'food', 'meal', 'snack', 'cuisine', 'restaurant']
        return self.__bundle(prefix, self.entities['CUISINE'], suffix)

    def __getlocation(self):
        prefix = ['', 'in ']
        return self.__bundle(prefix, self.entities['LOCATION'])
    
    def __getresttype(self):
        prefix = ['', 'in a ']
        suffix = ['', ' one', ' range', ' price range', ' restaurant']
        return self.__bundle(prefix, self.entities['REST_TYPE'], suffix)


    def getgreeting(self, entities):
        return {'RESPONSE': random.choice(self.greeting)}
    
    def getpeople(self, entities):
        return {'RESPONSE': self.__bundle(self.requesting, [self.__getpeople()], self.polite)}

    def getcuisine(self, entities):
        return {'RESPONSE': self.__bundle(self.requesting, [self.__getcuisine()], self.polite)}
    
    def getlocation(self, entities):
        return {'RESPONSE': self.__bundle(self.requesting, [self.__getlocation()], self.polite)}
    
    def getresttype(self, entities):
        return {'RESPONSE': self.__bundle(self.requesting, [self.__getresttype()], self.polite)}

    def getrequest(self, entities):
        reply = random.shuffle[random.choice[self.__getpeople(), ''], \
                random.choice[self.__getlocation(), ''], \
                random.choice[self.__getcuisine(), ''], \
                random.choice[self.__getresttype(), '']]
        return {'RESPONSE': self.__bundle(self.requesting, reply, self.polite)}

    def getadditional(self, entities):
        asking = ['Do you have', "Would you mind", "May I have", "What is", "Could you give me", 'Can you provide', ""]
        prefix = ['the', "it's", ""]
        types = ['phone', 'phone number', 'location', 'address', ""]
        suffix = ['of the restaurant']
        reply = self.__bundle(asking, prefix, types, suffix)
        reply = random.choice([random.choice(self.polite2), reply])
        return {'RESPONSE': reply} 
    

