#!/usr/bin/env python
from .backend import BackendBase
from sleekxmpp import ClientXMPP

class XMPPClient(ClientXMPP):
    def __init__(self, jid, password, session):
        ClientXMPP.__init__(self, jid, password)
        self.add_event_handler("session_start", self.session_start)
        self.add_event_handler("message", self.message)
        self.session = session

    def session_start(self, event):
        self.send_presence()
        self.get_roster()

    def message(self, msg):
        if msg['type'] in ('chat', 'normal'):
            question = msg["body"]
            from_client = msg["from"]
            reply, score = self.session(question, session_id=from_client)
            msg.reply(reply).send()


class XMPP(BackendBase):
    def __init__(self, session_config, jid, paassword, **args):
        super().__init__(session_config=session_config, **args)
        self.xmpp = XMPPClient(jid=jid, password=password, session=self.session)

    def run(self):
        #import logging
        #logging.basicConfig(level=logging.DEBUG, format='%(levelname)-8s %(message)s')
        self.xmpp.process(block=True)

