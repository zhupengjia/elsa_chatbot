#!/usr/bin/env python
from chatbot_end2end.module.interact_session import InteractSession
from nlptools.utils import Config

cfg = Config('config/elsa.yml')
s = InteractSession.build(cfg)
s.response("hello")
