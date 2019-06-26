#!/usr/bin/env python
from sleekxmpp import ClientXMPP

class XMPPClient(ClientXMPP):
    def __init__(self, jid, password):
        ClientXMPP.__init__(self, jid, password)
        self.add_event_handler("session_start", self.session_start)
        self.add_event_handler("message", self.message)
        
    def session_start(self, event):
        self.send_presence()
        self.get_roster()

    def message(self, msg):
        if msg['type'] in ('chat', 'normal'):
            question = msg["body"]
            from_client = msg["from"]
            reply = self.interact(question, from_client)
            msg.reply(reply).send()


class XMPP:
    def __init__(self, interact_session, cfg):
        self.xmpp = XMPPClient(jid=cfg.jid, password=cfg.password)

    def run(self):
        #import logging
        #logging.basicConfig(level=logging.DEBUG, format='%(levelname)-8s %(message)s')
        self.xmpp.process(block=True)

