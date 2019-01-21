#!/usr/bin/env python

from src.model.policy_gradiant import Policy_Gradiant
from src.hook.behaviors import Behaviors
from src.hook.babi_gensays import Babi_GenSays
from src.reader.reader_babi import Reader_Babi
from nlptools.utils import Config

cfg = Config("config/babi.yml")
hook = Behaviors()
ad_hook = Babi_GenSays(cfg.rule_based.hook_keywords) 
model = Policy_Gradiant.build(cfg, Reader_Babi, hook, ad_hook)
model.train()

