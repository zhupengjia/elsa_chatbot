#!/usr/bin/env python
import random, os, re
from .backend import BackendBase
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from nlptools.audio.speech import Speech_Deepspeech

class TelegramBackend(BackendBase):
    def __init__(self, session_config, token, deepspeech=None, tts=None, **args):
        super().__init__(session_config=session_config, **args)
        self.updater = Updater(token=token, use_context=True)
        self.dispatcher = self.updater.dispatcher
        self.dispatcher.add_handler(CommandHandler("start", TelegramBackend.start))
        self.dispatcher.add_handler(CommandHandler("reset", self.reset))
        self.dispatcher.add_handler(CommandHandler("tts", self.get_tts))
        self.dispatcher.add_handler(MessageHandler(Filters.text, self.query))
        self.dispatcher.add_handler(MessageHandler(Filters.voice, self.voice))
        if deepspeech:
            self.ds_model = Speech_Deepspeech(**deepspeech)
        else:
            self.ds_model = None
        if tts:
            from nlptools.audio.tts import MozillaTTS
            self.tts_model = MozillaTTS(**tts)
        else:
            self.tts_model = None

    @staticmethod
    def start(update, context):
        chat_id = update.message.chat_id
        context.bot.send_message(chat_id=chat_id, text="Elsa is here ~")

    def reset(self, update, context):
        chat_id = update.message.chat_id
        self.init_session()
        context.bot.send_message(chat_id=chat_id, text="Chatbot Reset")

    def get_tts(self, update, context):
        chat_id = update.message.chat_id
        text = update.message.text.strip()
        text = re.split("\s", text, maxsplit=1)[1]

        random_id = random.randint(0,100000)

        wav_base = "/tmp/{}_{}_%i.wav".format(chat_id, random_id)
        wav_file = "/tmp/{}_{}.wav".format(chat_id, random_id)
        ogg_file = "/tmp/{}_{}.ogg".format(chat_id, random_id)

        wav_files = []

        import spacy
        nlp = spacy.load("en")

        if self.tts_model:
            doc = nlp(text)
            for i, sent in enumerate(doc.sents):
                wavefile = wav_base%i
                wav_files.append(wavefile)
                text = sent.text.strip()
                self.tts_model(text, wavefile)
             
            sox_cmd = ["sox"] + wav_files + [wav_file]
            os.system(" ".join(sox_cmd))
            
            Speech_Deepspeech.wav2ogg(wav_file, ogg_file)
            with open(ogg_file, "rb") as oggf:
                context.bot.send_voice(chat_id=chat_id, voice=oggf)
            os.remove(wav_file)
            os.remove(ogg_file)
            for f in wav_files:
                os.remove(f)
        else:
            context.bot.send_message(chat_id=chat_id, text=text)

    def voice(self, update, context):
        chat_id = update.message.chat_id
        if not self.ds_model:
            context.bot.send_voice(chat_id=chat_id, voice=update.message.voice)
            return

        #convert voice to text
        random_id = random.randint(0,100000)
        ogg_file = "/tmp/{}_{}.ogg".format(chat_id, random_id)
        wav_file = "/tmp/{}_{}.wav".format(chat_id, random_id)
        response_wave_file = "/tmp/response_{}_{}.wav".format(chat_id, random_id)
        response_ogg_file = "/tmp/response_{}_{}.ogg".format(chat_id, random_id)

        update.message.voice.get_file().download(ogg_file)
        try:
            Speech_Deepspeech.ogg2wav(ogg_file, wav_file)
        except Exception as err:
            context.bot.send_message(chat_id=chat_id, text=err)
            os.remove(ogg_file)
            os.remove(wav_file)
            return

        text = self.ds_model(wav_file)
        os.remove(ogg_file)
        os.remove(wav_file)

        #query
        response, score = self.session(text, session_id=chat_id)

        #tts
        if self.tts_model:
            self.tts_model(response, response_wave_file)
            Speech_Deepspeech.wav2ogg(response_wave_file, response_ogg_file)
            with open(response_ogg_file, "rb") as oggf:
                context.bot.send_voice(chat_id=chat_id, voice=oggf)
            os.remove(response_wave_file)
            os.remove(response_ogg_file)
        else:
            context.bot.send_message(chat_id=chat_id, text=response)

    def query(self, update, context):
        chat_id = update.message.chat_id
        text = update.message.text.strip()
        response, score = self.session(text, session_id=chat_id)
        context.bot.send_message(chat_id=chat_id, text=response)

    def run(self):
        import logging
        logging.basicConfig(level=logging.WARNING, format='%(levelname)-8s %(message)s')
        self.updater.start_polling()
