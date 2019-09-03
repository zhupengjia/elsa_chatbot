#!/usr/bin/env python
import torch, numpy, math, os, copy
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from nlptools.text.ner import NER
from nlptools.text.tokenizer import Tokenizer_BERT
from nlptools.text.sentiment import Sentiment
from nlptools.text.spellcheck import SpellCorrection
from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule
from tensorboardX import SummaryWriter
from ..module.topic_manager import TopicManager
from ..module.dialog_status import dialog_collate
from .. import reader as Reader, skills as Skills
from ..reader import ReaderXLSX

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''


class Supervised:
    def __init__(self, reader, skill_name, batch_size=100, num_workers=1, epochs=1000, optimizer="adam", weight_decay=0, learning_rate=0.001, momentum=0.9, warmup_steps=0, adam_epsilon=1e-8, loss_scale=0, fp16_opt_level='O1', max_grad_norm=1.0, saved_model="model.pt", device='cpu', gpu_ids=None, save_per_epoch=1, gradient_accumulation_steps=1, logging_steps=50, **tracker_args):
        """
            Supervised learning for chatbot

            Input:
                - reader: reader instance, see ..module.reader.*.py
                - batch_size: int, default is 100
                - num_workers: int, default is 1
                - epochs: int, default is 1000
                - weight_decay: int, default is 0
                - learning_rate: float, default is 0.001
                - saved_model: str, default is "model.pt"
                - device: string of torch device, default is "cpu"
                - gpu_ids: list of gpu ids, for multiple gpu training
                - see Tracker class for other more args
        """
        self.reader = reader
        topic_manager = self.reader.topic_manager
        self.skill = topic_manager.skills[skill_name]
        self.save_per_epoch = save_per_epoch

        self.saved_model = saved_model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.epochs = epochs
        self.optimizer_type = optimizer
        self.momentum = momentum
        self.adam_epsilon = adam_epsilon
        self.warmup_steps = warmup_steps
        self.fp16_opt_level = fp16_opt_level
        self.max_grad_norm = max_grad_norm
        self.loss_scale = loss_scale
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.logging_steps = logging_steps
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')
        self.gpu_ids = gpu_ids

        self.__init_tracker(**tracker_args)

    @classmethod
    def build(cls, config):
        """
            construct model from config

            Input:
                - config: configure dictionary
        """
        if "ner" in config:
            ner_config = config.ner
        else:
            ner_config = None
        tokenizer = Tokenizer_BERT(**config.tokenizer)
        vocab = tokenizer.vocab
        sentiment_analyzer = Sentiment()
        spellcheck = SpellCorrection(**config.spell)

        # reader and skill class
        if not hasattr(Reader, config.reader.wrapper):
            raise RuntimeError("Error!! Reader {} not implemented!".format(config.reader.wrapper))
        if not hasattr(Skills, config.skill.wrapper):
            raise RuntimeError("Error!! Skill {} not implemented!".format(config.skill.wrapper))
        skill_cls = getattr(Skills, config.skill.wrapper)
        reader_cls = getattr(Reader, config.reader.wrapper)

        topic_manager = TopicManager()
        response_params = copy.deepcopy(config.skill)
        
        if "dialogflow" in config.skill:
            dialogflow = ReaderXLSX(config.skill.dialogflow,
                                    tokenizer=tokenizer,
                                    ner_config=ner_config) 
            entities = dialogflow.entities
            response_params.pop('dialogflow')
        else:
            dialogflow = None
            entities = None
        response_params.pop('wrapper')

        # skill
        response = skill_cls(tokenizer=tokenizer, vocab=vocab,
                             dialogflow=dialogflow,
                             max_seq_len=config.reader.max_seq_len,
                             skill_name=config.skill.name,
                             **response_params)

        topic_manager.register(config.skill.name, response)

        if ner_config is not None and entities is not None:
            for k in ["keywords", "regex", "ner_name_replace"]:
                if entities[k]:
                    if not k in ner_config:
                        ner_config[k] = {}
                    ner_config[k].update(entities[k])
            for k in ["ner"]:
                if entities[k]:
                    if not k in ner_config:
                        ner_config[k] = []
                    ner_config[k] += entities[k]
        if ner_config is not None:
            ner = NER(**ner_config)
        else:
            ner = None
        
        # reader
        max_entity_types = config.model.max_entity_types if 'max_entity_types' in config.model else 1024
        reader = reader_cls(vocab=vocab, tokenizer=tokenizer,
                            ner=ner, topic_manager=topic_manager,
                            sentiment_analyzer=sentiment_analyzer,
                            spell_check=spellcheck,
                            max_seq_len=config.reader.max_seq_len,
                            max_entity_types=max_entity_types,
                            flat_mode=config.reader.flat_mode)
        reader.read(config.reader.train_data)

        return cls(reader=reader, skill_name=config.skill.name, **config.model)

    def __init_tracker(self, **args):
        """
        tracker
        """
        self.skill.init_model(saved_model=self.saved_model, device=str(self.device), gpu_ids=self.gpu_ids, **args)

        self.fp16 = False

        param_optimizer = list(self.skill.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]


        num_train_optimization_steps = int(len(self.reader) / self.batch_size) * self.epochs
        if self.optimizer_type.lower() == "adam":
            self.optimizer = optim.Adam(optimizer_grouped_parameters, lr=self.learning_rate, eps=self.adam_epsilon, weight_decay=self.weight_decay)
        elif self.optimizer_type.lower() == "sgd":
            self.optimizer = optim.SGD(optimizer_grouped_parameters, lr=self.learning_rate,
                                       momentum=self.momentum)
        elif self.optimizer_type.lower() in ["fp16", "adamw"]:
            from apex import amp
            self.optimizer = AdamW(optimizer_grouped_parameters,
                       lr=self.learning_rate,
                       eps=self.adam_epsilon)
            if self.optimizer_type.lower() == "fp16":
                self.fp16 = True
                self.skill.model, self.optimizer = amp.initialize(self.skill.model, self.optimizer, opt_level=self.fp16_opt_level)

        self.scheduler = WarmupLinearSchedule(self.optimizer, warmup_steps=self.warmup_steps, t_total=num_train_optimization_steps)
    
        # multi-gpu training (should be after apex fp16 initialization)
        if self.gpu_ids and len(self.gpu_ids) > 1:
            self.skill.model = torch.nn.DataParallel(self.skill.model)

        print('Optimizer: {} with learning_rate: {}'.format(self.optimizer_type, self.learning_rate))

        self.start_epoch = 0
        self.best_loss = 1e9

        # checkpoint
        if os.path.exists(self.saved_model):
            if "optimizer_type" in self.skill.checkpoint\
               and self.skill.checkpoint["optimizer_type"] == self.optimizer_type:
                self.optimizer.load_state_dict(self.skill.checkpoint["optimizer"])
            self.start_epoch = self.skill.checkpoint['epoch']
            self.best_loss = self.skill.checkpoint["loss"] if "loss" in self.skill.checkpoint else 1e9

        self.generator = DataLoader(self.reader, batch_size=self.batch_size, collate_fn=dialog_collate, shuffle=True, num_workers=self.num_workers)

    def train(self):
        if self.fp16:
            from apex import amp
        self.skill.model.train() # set train flag
        ave_loss = []
        global_steps = -1
        tb_writer = SummaryWriter()
        self.skill.model.zero_grad()
        for epoch in trange(self.start_epoch, self.epochs, desc="Epoch"):
            pbar = tqdm(self.generator, desc="Iteration")
            for step, d in enumerate(pbar):
                if self.fp16:
                    d.half()
                d.to(self.device)

                _, loss = self.skill.get_response(d)

                if math.isnan(loss.item()):
                    print("Warning!! nan loss!")
                    continue

                if self.gpu_ids and len(self.gpu_ids) > 1:
                    loss = loss.mean()
                if self.gradient_accumulation_steps > 1:
                    loss = loss/self.gradient_accumulation_steps

                pbar.set_description('loss:{}'.format(loss.item()))

                if self.fp16:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.max_grad_norm)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.skill.model.parameters(), self.max_grad_norm)

                if (step +1) % self.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    ave_loss.append(loss.item())

                    if self.logging_steps > 0 and global_steps % self.logging_steps == 0:
                        tb_writer.add_scalar('lr', self.scheduler.get_lr()[0], global_steps)
                        tb_writer.add_scalar('loss', loss, global_steps)

                # save
                if (global_steps+1)%self.save_per_epoch == 0 and len(ave_loss)>0 and numpy.nanmean(ave_loss) < self.best_loss:
                    self.best_loss = loss
                    state = {
                        'state_dict': self.skill.model.state_dict(),
                        'config_model': self.skill.model.config,
                        'epoch': epoch,
                        'loss': loss.item(),
                        'optimizer': self.optimizer.state_dict(),
                        'optimizer_type': self.optimizer_type
                    }
                    torch.save(state, self.saved_model)
                    ave_loss = []
                global_steps += 1


        tb_writer.close()
