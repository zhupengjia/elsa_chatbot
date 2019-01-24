#!/usr/bin/env python
import sys, torch, os
from src.reader.reader_babi import Reader_Babi
from src.module.dialog_status import Dialog_Status
from src.hook.behaviors import Behaviors
from src.model.supervised import Supervised
from nlptools.utils import Config

from src.backend.xmpp import XMPP as Backend

class InteractSession(Backend):
    def __init__(self):
        self.cfg = Config('config/babi.yml')
        self.cfg.model.dropout = 0 #no dropout while predict
        self.cfg.logger.loglevel_console=10
        #self.cfg.model.use_gpu = 0 #use gpu

        hook = Behaviors()
        
        self.model = Supervised.build(self.cfg, Reader_Babi, hook)
        self.model.tracker.eval()

        self.reader = self.model.reader
        self.logger = self.model.logger
        
        self.dialog_status = {}

    def interact(self, query, session_id="default") :
        if not session_id in self.dialog_status:
            self.dialog_status[session_id] = self.reader.new_dialog()

        if query in ["clear", "reset", "restart", "exit", "stop", "quit", "q"]:
            self.dialog_status[session_id] = self.reader.new_dialog()
            return "reset"

        if self.dialog_status[session_id].add_utterance(query) is None:
            return ""

        self.dialog_status[session_id].getmask()
        data = Dialog_Status.torch(self.reader.entity_dict, [self.dialog_status[session_id]], self.reader.max_seq_len, self.reader.max_entity_types, device=self.reader.device)

        if data is None:
            return ""
        
        y_prob = self.model.tracker(data)
        _, y_pred = torch.max(y_prob.data, 1)

        y_pred = int(y_pred.cpu().numpy()[-1])

        self.dialog_status[session_id].add_response(y_pred)

        entities = {}
        for eid in self.dialog_status[session_id].entity:
            entity = self.dialog_status[session_id].entity_dict.entity_namedict.inv[eid]
            value = self.dialog_status[session_id].entity_dict.entity_dict.inv[self.dialog_status[session_id].entity[eid]]
            entities[entity] = value


        response = self.reader.get_response(y_pred, entities)
       
        response += "\n========= debug =======\n" + str(self.dialog_status[session_id])

        if y_pred == len(self.reader):
            response += '\n======= ' + 'information got:'
            for e, v in enumerate(entities):
                response += "\n" + e + ':' + 'v'
            self.dialog_status[session_id] = self.reader.new_dialog()

        return response
    

session = InteractSession()
session.run()

