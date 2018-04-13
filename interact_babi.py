#!/usr/bin/env python
import sys, torch, os
from src.reader.reader_babi import Reader_Babi
from src.model.dialog_tracker import Dialog_Tracker
from src.module.dialog_status import Dialog_Status
from nlptools.utils import Config
import torch.optim as optim


class InteractiveSession():
    def __init__(self):
        self.cfg = Config('config/babi.yml')
        self.cfg.model.use_gpu = 0 #use cpu
        self.cfg.model.dropout = 0 #no dropout while predict

        self.reader = Reader_Babi(self.cfg)
        self.reader.build_responses()
        self.reader.responses.build_mask()

        self.tracker = Dialog_Tracker(self.cfg.model, self.reader.vocab, len(self.reader))
        self.tracker.network()

        if self.cfg.model.use_gpu:
            self.tracker.cuda(self.cfg.model.use_gpu-1)

        #load checkpoint
        checkpoint = torch.load(self.cfg.model['saved_model'], map_location={'cuda:0': 'cpu'})
        if os.path.exists(self.cfg.model['saved_model']):
            self.tracker.load_state_dict(checkpoint['model'])

    def interact(self):
        dialog_status = self.reader.new_dialog()
        #interaction loop
        while True:
            # get input from user
            u = input(':: ')
            u = u.strip()
            if len(u) < 1:
                continue
        
            # check if user wants to begin new session
            if u in ['clear', 'reset', 'restart']:
                dialog_status = self.reader.new_dialog()
        
            # check for exit command
            elif u in ['exit', 'stop', 'quit', 'q']:
                break
        
            else:
                if dialog_status.add_utterance(u) is None:
                    continue
                dialog_status.getmask()
                data = Dialog_Status.torch(self.cfg, self.reader.vocab, self.reader.entity_dict, [dialog_status])
                if data is None:
                    continue
                #print(dialog_status)
                y_prob = self.tracker(data)
                _, y_pred = torch.max(y_prob.data, 1)
                y_pred = int(y_pred.numpy()[-1])
                

                dialog_status.add_response(y_pred)

                entities = {}
                for eid in dialog_status.entity:
                    entity = dialog_status.entity_dict.entity_namedict.inv[eid]
                    value = dialog_status.entity_dict.entity_dict.inv[dialog_status.entity[eid]]
                    entities[entity] = value
                response = self.reader.get_response(y_pred, entities)
                
                print(response)
                if y_pred == len(self.reader):
                    print('======= ' + 'information got:')
                    for e, v in enumerate(entities):
                        print(e + ':' + 'v')
                    dialog_status = self.reader.new_dialog()
                    



if __name__ == '__main__':
    # create interactive session
    isess = InteractiveSession()
    # begin interaction
    isess.interact()

     

