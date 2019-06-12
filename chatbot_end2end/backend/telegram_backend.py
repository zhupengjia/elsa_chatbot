#!/usr/bin/env python
import telegram

class TelegramBackend:
    def __init__(self, interact_session):
        self.session = interact_session
        self.bot = telegram.Bot(token='661472039:AAHcCQ8S57L31hMR5CaW5OANYoR5k8zhk4U')
    
    def run(self):
        while True:
            query = input(":: ")
            response = self.session.response(query)
            print(response)

    def query(self, text, chat_id='676345402'):
        response = self.session.response(text)
        self.bot.send_message(chat_id=chat_id, text=response)

