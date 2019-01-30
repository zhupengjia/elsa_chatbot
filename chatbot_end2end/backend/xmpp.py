#!/usr/bin/env python
from sleekxmpp import ClientXMPP

class XMPP(ClientXMPP):
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


