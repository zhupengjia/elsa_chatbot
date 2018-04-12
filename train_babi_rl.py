#!/usr/bin/env python

from src.model.policy_gradiant import Policy_Gradiant
from src.hook.babi_gensays import Babi_GenSays
from src.reader.reader_babi import Reader_Babi

pg = Policy_Gradiant('config/babi.yml', Reader_Babi, Babi_GenSays)
pg.train()

