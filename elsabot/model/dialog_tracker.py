#!/usr/bin/env python
import torch
import ipdb
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence
from nlptools.zoo.encoders.transformer import TransformerEncoder

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

class DialogTracker(nn.Module):
    '''
        dialog tracker for end2end chatbot 

        Input:
            - skill_name: string, current skill name
            - encoder: sentence encoder instance from .sentence_encoder
            - Nresponses: number of available responses
            - kernel_num: int
            - kernel_size: int
            - max_entity_types: int
            - fc_responses: int for int list, default is 5
            - entity_layers: int, default is 2
            - num_hidden_layers: int, default is 1
            - dropout: float, default is 0.2
            
    '''
    def __init__(self, skill_name, num_responses, max_entity_types, shared_layers=None,
                 bert_model_name=None, vocab_size=30522, encoder_freeze=False,
                 encoder_hidden_layers=12, encoder_attention_heads=12, max_position_embeddings=512,
                 encoder_intermediate_size=1024, encoder_hidden_size=768,
                 entity_layers=2, entity_emb_dim=50, num_hidden_layers=1, hidden_size=300,
                 dropout=0, **args):
        super().__init__()
        if shared_layers is None or not "encoder" in shared_layers:
            self.encoder = TransformerEncoder(vocab_size=vocab_size,
                                              bert_model_name=bert_model_name,
                                              num_hidden_layers=encoder_hidden_layers,
                                              num_attention_heads=encoder_attention_heads,
                                              max_position_embeddings=max_position_embeddings,
                                              hidden_size=encoder_hidden_size,
                                              intermediate_size=encoder_intermediate_size,
                                              dropout=dropout)

            if shared_layers is not None:
                shared_layers["encoder"] = self.encoder
        else:
            self.encoder = shared_layers["encoder"]
        if encoder_freeze: self.encoder.freeze()
        
        self.config = {"encoder":self.encoder.config,
                       "decoder":{
                           "num_responses": num_responses,
                           "entity_layers": entity_layers,
                           "entity_emb_dim": entity_emb_dim,
                           "num_hidden_layers": num_hidden_layers,
                           "max_entity_types": max_entity_types,
                           "hidden_size": hidden_size}
                      }

        encoder_hidden_size = self.config["encoder"].hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.num_responses = num_responses
        self.response_key = '$TENSOR_RESPONSE_' + skill_name
        self.mask_key = '$TENSOR_RESPONSE_MASK_' + skill_name

        fc_entity_layers = []
        for i in range(entity_layers-1, ):
            fc_entity_layers.append(nn.Linear(max_entity_types, max_entity_types))
            fc_entity_layers.append(nn.ReLU())
            fc_entity_layers.append(nn.Dropout(dropout))
        fc_entity_layers.append(nn.Linear(max_entity_types, entity_emb_dim))
        fc_entity_layers.append(nn.ReLU())
        fc_entity_layers.append(nn.Dropout(dropout))
        self.fc_entity = nn.Sequential(*fc_entity_layers)

        self.fc_dialog = nn.Sequential(nn.Linear(encoder_hidden_size+ entity_emb_dim, hidden_size),
                                       nn.ReLU(),
                                       nn.Dropout(dropout))
        
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_hidden_layers, batch_first=True)
        self.fc_out = nn.Sequential(nn.ReLU(),
                                    nn.Dropout(dropout),
                                    nn.Linear(hidden_size, hidden_size),
                                    nn.ReLU(),
                                    nn.Dropout(dropout),
                                    nn.Linear(hidden_size, num_responses)
                                   )
        self.loss_function = nn.NLLLoss()
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)
        #self.loss_function = nn.BCEWithLogitsLoss()
        #self.sigmoid = nn.Sigmoid()

    def dialog_embedding(self, utterance, utterance_mask, entity):
        '''
            Model framework:
                - utterance_embedding + entityname_embedding + prev_response embedding -> linear

            Get dialog embedding from utterance, entity, response_prev

            Input:
                - utterance, entity, response_prev are from three related keys of dialog_status.torch output

            Output:
                - dialog embedding
        '''
        #utterance embedding
        sequence_output, pooled_output = self.encoder(utterance, attention_mask=utterance_mask)

        #entity name embedding
        entity = self.fc_entity(entity)
       
        #concat together and apply linear
        utter = torch.cat((pooled_output, entity), 1)
        
        emb = self.fc_dialog(utter)
        
        return emb


    def forward(self, dialogs, incre_state=None):
        '''
            Model framework:
                - dialogs -> dialog_embedding -> gru -> softmax*mask -> logsoftmax
            
            Input:
                - dialogs: output from dialog_status.torch
            
            output:
                - logsoftmax
        '''
        #first get dialog embedding
        pack_batch = dialogs['utterance'].batch_sizes

        dialog_emb = self.dialog_embedding(dialogs['utterance'].data, dialogs["utterance_mask"].data, dialogs['entity'].data)

        dialog_emb = PackedSequence(dialog_emb, pack_batch) #feed batch_size and pack to packedsequence
        
        #dialog embedding to gru as dialog tracker
        obj_id = id(self)
        if incre_state is not None and obj_id in incre_state and not self.training:
            hidden = incre_state[obj_id]
        else:
            bsz = pack_batch[0]
            hidden = dialog_emb.data.new_zeros(self.num_hidden_layers, bsz, self.hidden_size)

        gru_out, hidden = self.gru(dialog_emb, hidden)

        if incre_state is not None and not self.training:
            incre_state[obj_id] = hidden
        
        out = self.fc_out(gru_out.data)

        if self.training and self.response_key in dialogs:
            out = self.logsoftmax(out)
            y_true = dialogs[self.response_key].data
            loss = self.loss_function(out, y_true.squeeze(1))
            #y_true_onehot = torch.zeros_like(out)
            #y_true_onehot.scatter_(1, y_true, 1)
            #loss = self.loss_function(out, y_true_onehot.squeeze(1))
            return out, loss

        probs = self.softmax(out)
        #probs = self.sigmoid(out)
        #print(probs)

        #apply mask
        response = probs * dialogs[self.mask_key].data
        return response


