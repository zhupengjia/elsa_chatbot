#!/usr/bin/env python
from .backend import BackendBase
import werobot

class WechatClient(werobot.WeRoBot):
    def __init__(self, token, app_id, app_secret, host, port, **args):
        super().__init__(token=token, APP_ID=app_id, APP_SECRET=app_secret, HOST=host, PORT=port, enable_session=True)


    def reset(self):
        return "reset"


    def interact(self, query, session_id="wechat"):
        return "response"


class Wechat(BackendBase):
    def __init__(self, session_config, **args):
        super().__init__(session_config=session_config, **args)
        self.client = WechatClient(**args)
