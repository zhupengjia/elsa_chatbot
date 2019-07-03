#!/usr/bin/env python
from chatbot_end2end.module.interact_session import InteractSession
from nlptools.utils import Config

class BackendBase:
    def __init__(self, session_config, **args):
        self.config_path = session_config
        self.init_session()

    def init_session(self):
        cfg = Config(self.config_path)
        self.session = InteractSession.build(cfg)

class Backend:
    def __new__(cls, session, backend_type="shell", **args):
        if backend_type in ["restful", "restapi"]:
            from .restful import Restful
            return Restful(session, **args)
        elif backend_type == "xmpp":
            from .xmpp import XMPP
            return XMPP(session, **args)
        elif backend_type == "wechat":
            from .wechat import Wechat
            return Wechat(session, **args)
        elif backend_type == "telegram":
            from .telegram_backend import TelegramBackend
            return TelegramBackend(session, **args)
        else:
            from .shell import Shell
            return Shell(session)
