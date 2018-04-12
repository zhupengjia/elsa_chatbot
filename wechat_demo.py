#!/usr/bin/env python
import sys, os, werobot


class WechatSession(werobot.WeRoBot):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def interact(self, u, sessionid):
        #interaction loop
        u = u.strip()
        if len(u) < 1:
            return ':)'
        return u    

#wechat
robot = WechatSession(enable_session=True,\
    token='EGTqL3cPdY', \
    APP_ID='wxb43165167096f77d',\
    APP_SECRET='d646417bf85eed8fbf72a40b106c328e',\
    HOST='127.0.0.1',\
    PORT='8888'\
    )



@robot.text
def session(message):
    # get input from user
    msg = message.content
    t = message.time
    user = message.source
    return robot.interact(msg, user)
                

robot.run()



     

