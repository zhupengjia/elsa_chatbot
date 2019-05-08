#!/usr/bin/env python


"""
    Author: Pengjia Zhu (zhupengjia@gmail.com)
"""

class SimilarityBase:
    def __init__(self):
        pass

    def eval(self):
        pass

    def to(self, device):
        pass


class BERTSim(SimilarityBase):
    """
        Similarity via BERT embedding

        Input:
            - encoder: BERT encoder
    """
    def __init__(self, encoder, device="cpu", **args):
        self.encoder = encoder
        self.device = device
    
    @property
    def dim(self):
        '''
            dimention of sentence embedding
        '''
        return self.encoder.config.hidden_size

    def eval(self):
        self.encoder.eval()

    def to(self, device):
        self.encoder.to(device)

    def get_embedding(self, sentences, sentence_masks):
        '''
            return sentence embedding

            Input:
                - sentences: list of sentence ids
                - sentence_masks: list of sentence masks
        '''

        sentence = numpy.concatenate(sentence)
        sentence_masks = numpy.concatenate(sentence_masks)
        sentence_ids = torch.LongTensor(sentence_ids).to(self.device)
        attention_mask = torch.LongTensor(attention_mask).to(self.device)
        sequence_output, pooled_output = self.encoder(sentence_ids, attention_mask=attention_mask, output_all_encoded_layers=False)
        return pooled_output.cpu().detach().numpy()



