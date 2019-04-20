#!/usr/bin/env python
import torch, math, sys
import torch.nn as nn
from nlptools.zoo.encoders.transformer_decoder import TransformerDecoder
from .sequence_generator import SequenceGenerator

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

class Generative_Tracker(nn.Module):
    '''
        Generative based chatbot

        Input:
            - skill_name: string, current skill name
            - encoder: sentence encoder instance from .sentence_encoder
            - **generator_args: arguments to sequence generator
    '''
    def __init__(self, skill_name, encoder, decoder_hidden_layers=1, decoder_attention_heads=2, decoder_hidden_size=1024, dropout=0, bos_id=1, eos_id=2, max_seq_len=100, beam_size=1, **args):
        super().__init__()
        self.config = {
                    "bert_model_name": encoder.bert_model_name,
                    "decoder_hidden_layers": decoder_hidden_layers,
                    "decoder_attention_heads": decoder_attention_heads,
                    "decoder_hidden_size": decoder_hidden_size
                }

        self.response_key = 'response_' + skill_name
        self.mask_key = 'response_mask_' + skill_name
        self.encoder = encode
        embedding_dim = self.encoder.embedding.word_embeddings.embedding_dim 
        self.num_embeddings = self.encoder.embedding.word_embeddings.num_embeddings
        self.control_layer = nn.Linear(embedding_dim+1, embedding_dim)
        self.decoder = TransformerDecoder(self.encoder.embedding, decoder_hidden_layers, decoder_attention_heads, decoder_hidden_size, dropout)
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.beam_size = beam_size
        self.max_seq_len = max_seq_len
        self.loss_function = nn.NLLLoss(ignore_index=0) #ignore padding loss
        self.logsoftmax = nn.LogSoftmax(dim=2)


    def dialog_embedding(self, utterance, utterance_mask, sentiment):
        #utterance embedding
        sequence_output, pooled_output = self.encoder(utterance, attention_mask=utterance_mask, output_all_encoded_layers=False)

        #sentiment
        sentiment = sentiment.unsqueeze(1).expand(-1, sequence_output.size(1), 1)

        #combine together
        sequence_out = torch.cat((sequence_output, sentiment), 2)
        sequence_out = self.control_layer(sequence_out)

        return sequence_out

    def beam_search(self, encoder_out, utterance_mask):
        bsz = encoder_out.size(0)
        src_len = encoder_out.size(1)
        # initialize buffers
        scores = encoder_out.new_zeros(bsz * self.beam_size, self.max_seq_len+1)
        scores_buf = scores.clone()
        tokens = encoder_out.new_zeros(bsz * self.beam_size, self.max_seq_len + 2).long()
        tokens_buf = tokens.clone()
        tokens[:, 0] = self.bos_id

    def forward(self, dialogs):
        #encoder
        utterance_mask = dialogs["utterance_mask"].data
        encoder_out = self.dialog_embedding(dialogs['utterance'].data, utterance_mask, dialogs["sentiment"].data)

        #decoder
        if self.training:
            prev_output = dialogs[self.response_key].data[:, :-1]
            target_output = dialogs[self.response_key].data[:, 1:]
           
            target_output = target_output.unsqueeze(-1).contiguous().view(-1)
            output, attn = self.decoder(prev_output, encoder_out, utterance_mask)
            output_probs = self.logsoftmax(output)

            output_probs_expand = output_probs.contiguous().view(-1, output_probs.size(2))

            loss = self.loss_function(output_probs_expand, target_output)
            return output_probs, loss
        else:
            self.beam_search(encoder_out, utterance_mask)
            

            incremental_state = {}
            
            print(output_buf.shape)
            print(output_buf[:,:1].shape)
            print(output_buf)

            output, attn = self.decoder(output_buf[:,:1], encoder_out, utterance_mask, 
                                        time_step=0, incremental_state=incremental_state)
            output_probs = self.logsoftmax(output)

            print(encoder_out.shape)
            print(output_buf.shape)
            print(scores_buf.shape)
            print(output_probs)

            sys.exit()

        sys.exit()

        return output_probs

