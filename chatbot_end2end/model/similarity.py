#!/usr/bin/env python


"""
    Author: Pengjia Zhu (zhupengjia@gmail.com)
"""

class SimilarityBase:
    def __init__(self):
        pass


class BERTSim(SimilarityBase):
    """
        Similarity via BERT embedding  
    """
    def __init__(self, bert_encoder):
        self.encoder = bert_encoder
    
    @property
    def dim(self):
        '''
            dimention of sentence embedding
        '''
        return self.encoder.config.hidden_size

    def eval(self):
        self.encoder.eval()

    def get_embedding(self, sentences, sentence_masks):
        '''
            return sentence embedding

            Input:
                - sentences: string or ids
                - batch_size: int, default is 100
        '''
        sequence_output, pooled_output = self.encoder(sentence_ids, attention_mask=attention_mask, output_all_encoded_layers=False)
        return pooled_output.cpu().detach().numpy()



