#!/usr/bin/env python
import sys, torch, os, werobot
from src.reader.reader_babi import Reader_Babi
from src.model.dialog_tracker import Dialog_Tracker
from nlptools.utils import Config
import torch.optim as optim


#cfg = Config('config/babi.yml')
#if not torch.cuda.is_available(): cfg.model.use_gpu = 0
#cfg.model.dropout = 0 #no dropout while predict
#
#reader = Reader_Babi(cfg)
#reader.build_responses()
#reader.responses.build_mask()
#
#tracker = Dialog_Tracker(cfg.model, reader.vocab, len(reader))
#tracker.network()
#
#if cfg.model.use_gpu:
#    tracker.cuda(cfg.model.use_gpu-1)
#
##load checkpoint
#checkpoint = torch.load(cfg.model['saved_model'], map_location={'cuda:0': 'cpu'})
#if os.path.exists(cfg.model['saved_model']):
#    tracker.load_state_dict(checkpoint['model'])
#
#wechat
robot = werobot.WeRoBot(token='d646417bf85eed8fbf72a40b106c328e')

@robot.text
def session(message, session):
    last = session.get("last", None)
    if last:
        return last
    session["last"] = message.content
    return '这是你第一次和我说话'

robot.run()


#    dialog_status = reader.new_dialog()
    #interaction loop
#def interact(u):
#    # get input from user
#    u = u.strip()
#    if len(u) < 1:
#        continue
#
#    # check if user wants to begin new session
#    if u in ['clear', 'reset', 'restart']:
#        dialog_status = reader.new_dialog()
#
#    # check for exit command
#    elif u in ['exit', 'stop', 'quit', 'q']:
#        break
#
#    else:
#        data = dialog_status(u)
#        if data is None:
#            continue
#        #print(dialog_status)
#        y_prob = tracker(data['utterance'], data['entity'], data['mask'])
#        _, y_pred = torch.max(y_prob.data, 1)
#        y_pred = y_pred.numpy()[0]
#        response = reader.get_response(y_pred)
#        dialog_status.add_response(y_pred)
#        print(response)
#        if y_pred == 11:
#            print('======= ' + 'information got:')
#            for eid in dialog_status.entity:
#                entity = dialog_status.entity_dict.entity_namedict.inv[eid]
#                value = dialog_status.entity_dict.entity_dict.inv[dialog_status.entity[eid]]
#                print(entity + ': ' + value)
#            dialog_status = reader.new_dialog()
                




     

