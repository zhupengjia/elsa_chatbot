#!/usr/bin/env python
import torch, math, numpy, sys
import torch.nn as nn
from nlptools.zoo.encoders.transformer import TransformerDecoder
from .sentence_encoder import Sentence_Encoder

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

class Generative_Tracker(nn.Module):
    '''
        Generative based chatbot

        Input:
            - skill_name (string): current skill name
            - shared_layers: dictionary, layers to share between networks
            - decoder_hidden_layers (int, optional): number of decoder layers, used for training only, default is 1
            - decoder_attention_heads (int, optional): number of decoder heads, used for training only, default is 2
            - decoder_hidden_size (int, optional): decoder hidden size, used for training only, default is 1024
            - dropout (float, option): dropout, default is 0
            - pad_id (int, option): PAD ID, used for prediction only, default is 0
            - bos_id (int, option): BOS ID, used for prediction only, default is 1
            - eos_id (int, option): EOS ID, used for prediction only, default is 2
            - unk_id (int, option): UNK ID, used for prediction only, default is 3
            - beam_size (int, optional): beam width, used for prediction only, default is 1
            - len_penalty (float, optional): length penalty, where < 1.0 favors shorter, >1.0 favors longer sentences. Used for prediction only, Default is 1.0
            - unk_penalty (float, optional): unknown word penalty, where < 1.0 favors more unks, >1.0 favors less. Used for prediction only. default is 1.0
    '''
    def __init__(self, skill_name, shared_layers=None, bert_model_name="bert-base-uncased", decoder_hidden_layers=1, decoder_attention_heads=2, decoder_hidden_size=1024, dropout=0, pad_id=0, bos_id=1, eos_id=2, unk_id=3, beam_size=1, len_penalty=1., unk_penalty=1., **args):
        super().__init__()
        if shared_layers is None or not "encoder" in shared_layers:
            self.encoder = Sentence_Encoder(bert_model_name)
            if shared_layers is not None:
                shared_layers["encoder"] = self.encoder
        else:
            self.encoder = shared_layers["encoder"]

        self.config = {
                    "bert_model_name": self.encoder.bert_model_name,
                    "decoder_hidden_layers": decoder_hidden_layers,
                    "decoder_attention_heads": decoder_attention_heads,
                    "decoder_hidden_size": decoder_hidden_size
                }

        self.response_key = 'response_' + skill_name
        self.mask_key = 'response_mask_' + skill_name

        embedding_dim = self.encoder.embedding.word_embeddings.embedding_dim
        self.num_embeddings = self.encoder.embedding.word_embeddings.num_embeddings
        self.control_layer = nn.Linear(embedding_dim+1, embedding_dim)
        self.decoder = TransformerDecoder(self.encoder.embedding, decoder_hidden_layers, decoder_attention_heads, decoder_hidden_size, dropout)
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.unk_id = unk_id
        self.beam_size = beam_size
        self.len_penalty = len_penalty
        self.unk_penalty = unk_penalty
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
        max_len = encoder_out.size(1)

        # initialize buffers
        scores_buf = encoder_out.new_zeros(bsz * self.beam_size, max_len)
        output_buf = encoder_out.new_zeros(bsz * self.beam_size, max_len).long()
        output_buf[:, 0] = self.bos_id
        finalized = utterance_mask.new_zeros(bsz * self.beam_size, max_len).byte()

        # expand encoder out and utterance mask
        encoder_out = encoder_out.repeat(self.beam_size, 1, 1)
        utterance_mask = utterance_mask.repeat(self.beam_size, 1)

        incre_state = {}

        for i in range(max_len - 1):
            # print("="*20)
            # print("output_buf", output_buf[:, i:i+1].size())
            output = self.decoder(output_buf[:,i:i+1], encoder_out, utterance_mask, 
                                    time_step=i, incre_state=incre_state)

            output_probs = self.logsoftmax(output)
            output_probs = output_probs.contiguous().view(-1, output_probs.size(2))

            # print("output_probs", output_probs.size())

            # prob union with previous outputs, normalized with sentence length
            prev_scores = scores_buf[:, i:i+1].expand(-1, output_probs.size(1)) 

            output_probs = (output_probs + prev_scores)/2 

            # remove special characters
            output_probs[:, self.pad_id] = -1e9
            output_probs[:, self.bos_id] = -1e9

            # add unk penalty
            output_probs[:, self.unk_id] *= self.unk_penalty

            # force set score of finalized sequence to previous sequence, with length penalty
            output_probs[finalized[:, i], :] = -1e9
            output_probs[finalized[:, i], 0] =\
                    scores_buf[finalized[:, i], i] * self.len_penalty 
            
            # get maximum probs 
            output_probs = output_probs.view(bsz, -1)
            
            output_max = output_probs.topk(self.beam_size)
           
            output_max_current = output_max[1].fmod(self.num_embeddings)
            output_max_prev = output_max[1].div(self.num_embeddings).long()

            output_max_current = output_max_current.view(-1)
            output_max_prev = output_max_prev.view(-1)

            #reorder previous outputs
            for j in range(bsz):
                output_buf[j*self.beam_size:(j+1)*self.beam_size, :i+1] =\
                        output_buf[j*self.beam_size:(j+1)*self.beam_size, :i+1][output_max_prev, :]
                scores_buf[j*self.beam_size:(j+1)*self.beam_size, :i+1] =\
                        scores_buf[j*self.beam_size:(j+1)*self.beam_size, :i+1][output_max_prev, :]
                finalized[j*self.beam_size:(j+1)*self.beam_size, :i+1] =\
                        finalized[j*self.beam_size:(j+1)*self.beam_size, :i+1][output_max_prev, :]
           
            self.decoder.reorder_incremental_state(incre_state, output_max_prev) 

            output_buf[:, i+1] = output_max_current
            scores_buf[:, i+1] = output_max[0].view(-1)
            
            finalized[:, i+1] = finalized[:, i] & output_max_current.eq(self.eos_id)

            if finalized[:, i+1].all():
                break

        # get maximum prob from last_score
        output = encoder_out.new_zeros(bsz, max_len).long()
        scores = numpy.zeros(bsz)
        for j in range(bsz):
            maxid = torch.argmax(scores_buf[j*self.beam_size:(j+1)*self.beam_size, i+1]).item()
            output[j, :] = output_buf[j*self.beam_size+maxid, :]
            scores[j] = scores_buf[j*self.beam_size+maxid, i+1]
        
        #print("scores_buf", scores_buf)
        #print("output_buf", output_buf)

        return output, scores

    def forward(self, dialogs):
        #encoder
        utterance_mask = dialogs["utterance_mask"].data
        encoder_out = self.dialog_embedding(dialogs['utterance'].data, utterance_mask, dialogs["sentiment"].data)

        #decoder
        if self.training:
            prev_output = dialogs[self.response_key].data[:, :-1]
            target_output = dialogs[self.response_key].data[:, 1:]
           
            target_output = target_output.unsqueeze(-1).contiguous().view(-1)
            
            output = self.decoder(prev_output, encoder_out, utterance_mask)
            output_probs = self.logsoftmax(output)

            output_probs_expand = output_probs.contiguous().view(-1, output_probs.size(2))

            loss = self.loss_function(output_probs_expand, target_output)
            return output_probs, loss
        else:
            return self.beam_search(encoder_out, utterance_mask)

