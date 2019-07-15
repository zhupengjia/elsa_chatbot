#!/usr/bin/env python
from .backend import BackendBase
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

class TelegramBackend(BackendBase):
    def __init__(self, session_config, token, **args):
        super().__init__(session_config=session_config, **args)
        self.updater = Updater(token=token, use_context=True)
        self.dispatcher = self.updater.dispatcher
        self.dispatcher.add_handler(CommandHandler("start", TelegramBackend.start))
        self.dispatcher.add_handler(CommandHandler("reset", self.reset))
        self.dispatcher.add_handler(MessageHandler(Filters.text, self.query))

    @staticmethod
    def start(update, context):
        chat_id = update.message.chat_id
        context.bot.send_message(chat_id=chat_id, text="Elsa is here ~")
    
    def reset(self, update, context):
        chat_id = update.message.chat_id
        self.init_session()
        context.bot.send_message(chat_id=chat_id, text="Chatbot Reset")

    def query(self, update, context):
        chat_id = update.message.chat_id
        text = update.message.text.strip()
        response, score = self.session(text, session_id=chat_id)
        context.bot.send_message(chat_id=chat_id, text=response)

    def run(self):
        import logging
        logging.basicConfig(level=logging.WARNING, format='%(levelname)-8s %(message)s')
        self.updater.start_polling()
