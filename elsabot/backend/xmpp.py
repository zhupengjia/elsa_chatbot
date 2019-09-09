#!/usr/bin/env python
from ..module.interact_session import InteractSession
from nlptools.utils import Config
from slixmpp import ClientXMPP

class XMPPClient(ClientXMPP):
    def __init__(self, jid, password, session_config):
        ClientXMPP.__init__(self, jid, password)
        self.jabber_id = jid
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
                session_id, reply, score = self.session(question, session_id=from_client)
                if isinstance(reply, dict):
                    for sid in reply:
                        if sid == from_client:
                            msg.reply(reply[sid]).send()
                        elif sid != self.jabber_id:
                            self.send_message(mto=sid, mbody=reply[sid])
                else:
                    if session_id == from_client:
                        msg.reply(reply).send()
                    elif session_id != self.jabber_id:
                        self.send_message(mto=session_id, mbody=reply)

        else:
            #TODO, other type of messages
            msg.reply(msg["type"]).send()


class XMPP:
    def __init__(self, session_config, jid, password, host, port=5222, proxy_host=None, proxy_port=None, **args):
        self.xmpp = XMPPClient(jid=jid, password=password, session_config=session_config)
        self.xmpp.register_plugin('xep_0030') # Service Discovery
        self.xmpp.register_plugin('xep_0004') # Data Forms
        self.xmpp.register_plugin('xep_0060') # PubSub
        self.xmpp.register_plugin('xep_0199') # XMPP Ping

        self.host = host
        self.port = port
        self.proxy_host = proxy_host
        self.proxy_port = proxy_port

    def run(self):
        import logging
        logging.basicConfig(level=logging.WARNING, format='%(levelname)-8s %(message)s')
        if self.proxy_host:
            self.xmpp.use_proxy=True
            self.xmpp.proxy_config = {
                'host': self.proxy_host,
                'port': self.proxy_port
            }
        self.xmpp.connect((self.host, self.port))
        self.xmpp.process(forever=True)

