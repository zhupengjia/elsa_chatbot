#!/usr/bin/env python3
import sys, torch, os
from ailab.utils import Config, setLogger
import torch.nn as nn
import torch.optim as optim
from src.reader.quora_reader import QuoraReader
from src.model.dulicate_embedding import Duplicate_Embedding

use_gpu = 0 if torch.cuda.is_available() else 0
cfg = Config('config/quora.yaml')
logger = setLogger(cfg)



