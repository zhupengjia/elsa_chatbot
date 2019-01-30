#!/usr/bin/env python
import werobot

class Wechat(werobot.WeRoBot):
    def __init__(self, token, app_id, app_secret, host='127.0.0.1', port=8888, **args):
        super().__init__(token=token, APP_ID=app_id, APP_SECRET=app_secret, HOST=host, PORT=port, enable_session=True, **args)


    def reset(self):
        return "reset"

    
    def interact(self, query, session_id="wechat"):
        return "response"

    


