#!/usr/bin/env python
import sys, torch, os, werobot
from src.reader.reader_babi import Reader_Babi
from src.model.dialog_tracker import Dialog_Tracker
from nlptools.utils import Config
import torch.optim as optim



class WechatSession(werobot.WeRoBot):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cfg = Config('config/babi.yml')
        if not torch.cuda.is_available(): self.cfg.model.use_gpu = 0
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

        self.dialog_status = {}

    def interact(self, u, sessionid):
        if not sessionid in self.dialog_status:
            self.dialog_status[sessionid] = self.reader.new_dialog()
        #interaction loop
        u = u.strip()
        if len(u) < 1:
            return ':)'
        
        # check if user wants to begin new session
        if u in ['clear', 'reset', 'restart', 'exit', 'stop', 'quit', 'q']:
            self.dialog_status[sessionid] = self.reader.new_dialog()
            return 'reset the dialog'

        else:
            data = self.dialog_status[sessionid](u)
            if data is None:
                return ':)'
            y_prob = self.tracker(data['utterance'], data['entity'], data['mask'])
            _, y_pred = torch.max(y_prob.data, 1)
            y_pred = y_pred.numpy()[0]
            response = self.reader.get_response(y_pred)
            self.dialog_status[sessionid].add_response(y_pred)
            if y_pred == 11:
                response += '\n======= ' + 'information got:'
                for eid in self.dialog_status[sessionid].entity:
                    entity = self.dialog_status[sessionid].entity_dict.entity_namedict.inv[eid]
                    value = self.dialog_status[sessionid].entity_dict.entity_dict.inv[self.dialog_status[sessionid].entity[eid]]
                    response += '\n' + entity + ': ' + value
                self.dialog_status[sessionid] = self.reader.new_dialog()
            return response


#wechat
robot = WechatSession(enable_session=True,\
    token='EGTqL3cPdY', \
    APP_ID='wxb43165167096f77d',\
    APP_SECRET='d646417bf85eed8fbf72a40b106c328e',\
    HOST='127.0.0.1',\
    PORT='8888'\
    )



@robot.text
def session(message):
    # get input from user
    msg = message.content
    t = message.time
    user = message.source
    return robot.interact(msg, user)
                

robot.run()



     

