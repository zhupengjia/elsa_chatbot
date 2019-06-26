#!/usr/bin/env python
import werobot

class Wechat(werobot.WeRoBot):
    def __init__(self, session, cfg):
        super().__init__(token=cfg.token, APP_ID=cfg.app_id, APP_SECRET=cfg.app_secret, HOST=cfg.host, PORT=cfg.port, enable_session=True)


    def reset(self):
        return "reset"

    
    def interact(self, query, session_id="wechat"):
        return "response"

    


