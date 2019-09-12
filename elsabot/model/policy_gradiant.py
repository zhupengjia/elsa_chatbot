#!/usr/bin/env python
import torch, numpy, math, os, copy
import torch.optim as optim
import ipdb
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule
from tensorboardX import SummaryWriter
from ..module.topic_manager import TopicManager
from ..module.dialog_status import dialog_collate
from ..module.interact_session import InteractSession
from .. import reader as Reader, skills as Skills
from ..reader import ReaderXLSX
from .supervised import Supervised

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''


class PolicyGradiant(Supervised):
    def __init__(self, simulator, maxloop=20, discount=0.95, **args):
        """
            Policy Gradiant for chatbot

            Input:
                - simulator: user simulator chatbot
        """
        self.simulator = simulator
        self.simulator_id = "USERSIMULATOR"
        self.baseline = 0
        self.maxloop = maxloop
        self.discount = discount
        super().__init__(**args)

    @classmethod
    def build(cls, config, **args):
        """
            construct model from config

            Input:
                - config: configure dictionary

        """
        # user simulator
        simulator = InteractSession.build(config.simulator)
        return super().build(simulator=simulator, config=config)

    def __memory_replay(self):
        """memory replay"""
        dialogs = []
        N_true = 0
        for batch in range(self.batch_size):
            # reset adversal chatbot
            self.simulator.reset(self.simulator_id)
            # greeting utterance
            utterance = self.simulator('', self.simulator_id)
            # new dialog status
            dialog_status = self.reader.new_dialog()
            # dialog loop
            for loop in range(self.maxloop):
                dialog_status.add_utterance(utterance[1])
                response, score = dialog_status.get_response(response_sentiment=0, device=self.device)

                if dialog_status.current_status["$SESSION_RESET"]:
                    N_true += 1
                    break

                # get new utterance
                utterance = self.simulator(response, self.simulator_id)
                ipdb.set_trace()
            dialogs.append(dialog_status) 

        # convert to torch variable
        dialogs_reward = [d.reward(self.baseline) for d in dialogs]
        dialogs_data = [d.data() for d in dialogs]
        dialogs_data = dialog_collate([[torch.tensor(d[k]) for k in d] for d in dialogs_data])
        return dialogs_data, float(N_true)/self.reader.batch_size

    def train(self):
        """
            Train the model. No input needed
        """
        self.skill.model.zero_grad()
        tb_writer = SummaryWriter()
        for epoch in trange(self.epochs, desc="Epoch"):
            self.skill.model.eval() # set eval flag
            dialogs, precision = self.__memory_replay()
            self.skill.model.train() # set train flag

            self.tracker.zero_grad()
            y_prob = self.tracker(dialogs)

            m = torch.distributions.Categorical(y_prob)
            action = m.sample()

            loss = -m.log_prob(action) * dialogs['reward']

            loss.sum().backward()
            self.optimizer.step()

            if epoch > 0 and epoch%10 == 0: 
                model_state_dict = self.tracker.state_dict()
                torch.save(model_state_dict, self.saved_model)

