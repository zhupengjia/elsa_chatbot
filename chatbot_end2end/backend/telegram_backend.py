#!/usr/bin/env python
from .backend import BackendBase
import telegram

class TelegramBackend(BackendBase):
    def __init__(self, session_config, token, **args):
        super().__init__(session_config=session_config, **args)
        self.bot = telegram.Bot(token=token)

    def query(self, text, chat_id='676345402'):
        text = text.strip()
        if text in ["reset"]:
            self.init_session()
            self.bot.send_message(chat_id=chat_id, text="reset all")
            return
        response = self.session.response(text)
        self.bot.send_message(chat_id=chat_id, text=response)

    def run(self):
        self.bot.start_polling()
