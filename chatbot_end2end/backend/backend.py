#!/usr/bin/env python
from .shell import Shell

class Backend:
    def __new__(cls, session, cfg=None):
        if cfg:
            if cfg.type == "restful":
                from .restful import Restful
                return Restful(session, cfg)
            elif cfg.type == "xmpp":
                from .xmpp import XMPP
                return XMPP(session, cfg)
            elif cfg.type == "wechat":
                from .wechat import Wechat
                return Wechat(session, cfg)
            elif cfg.type == "telegram":
                from .telegram_backend import TelegramBackend
                return TelegramBackend(session, cfg)
        return Shell(session)
