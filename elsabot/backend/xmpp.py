#!/usr/bin/env python
from ..module.interact_session import InteractSession
from nlptools.utils import Config
from sleekxmpp import ClientXMPP

class XMPPClient(ClientXMPP):
    def __init__(self, jid, password, session_config):
        ClientXMPP.__init__(self, jid, password)
        self.add_event_handler("session_start", self.session_start)
        self.add_event_handler("message", self.message)
        self.config_path = session_config
        self.init_session()
    
    def init_session(self):
        cfg = Config(self.config_path)
        self.session = InteractSession.build(cfg)

    def session_start(self, event):
        self.send_presence()
        self.get_roster()

    def message(self, msg):
        if msg['type'] in ('chat', 'normal'):
            question = msg["body"]
            from_client = msg["from"]
            if question in ["reset"]:
                self.init_session()
                msg.reply("reset all").send()
            else:
                reply, score = self.session(question, session_id=from_client)
                msg.reply(reply).send()
        else:
            print(msg)
            msg.reply(msg["type"]).send()


class XMPP:
    def __init__(self, session_config, jid, password, host, port=5222, **args):
        self.xmpp = XMPPClient(jid=jid, password=password, session_config=session_config)
        self.host = host
        self.port = port

    def run(self):
        import logging
        logging.basicConfig(level=logging.WARNING, format='%(levelname)-8s %(message)s')
        self.xmpp.connect((self.host, self.port))
        self.xmpp.process(block=True)

