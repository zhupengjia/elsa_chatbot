#!/usr/bin/env python
import telegram

class TelegramBackend:
    def __init__(self, interact_session, cfg):
        self.session = interact_session
        self.bot = telegram.Bot(token=cfg.token)
    
    def run(self):
        while True:
            query = input(":: ")
            response = self.session(query)
            print(response)

    def query(self, text, chat_id='676345402'):
        response = self.session.response(text)
        self.bot.send_message(chat_id=chat_id, text=response)

