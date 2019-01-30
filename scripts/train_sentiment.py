#!/usr/bin/env python
import glob, numpy, gzip, os, json, torch, sys
sys.path.append("..")
from nlptools.text  import Tokenizer
import torch.nn as nn
from chatbot_end2end.model.sentence_encoder import Sentence_Encoder
from chatbot_end2end.model.sentiment import Sentiment
import torch.optim as optim


class DataReader:
    def __init__(self, data_folder, bert_model_name, max_seq_len=30, batch_size=50):
        self.dataloc = glob.glob(os.path.join(data_folder, "*.json.gz"))
        self.tokenizer = Tokenizer(tokenizer='bert', bert_model_name=bert_model_name)
        self.vocab = self.tokenizer.vocab
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
   
    def jsonconvert(self, jsonstr):
        data = json.loads(jsonstr)
        text = data['reviewText']
        rate = int(data["overall"]) - 1
        text_ids = numpy.zeros(self.max_seq_len)
        text_mask = numpy.zeros(self.max_seq_len)

        temp_ids = self.vocab(self.tokenizer(text))[:self.max_seq_len-2]
        seq_len = len(temp_ids) + 2
        
        text_ids[0] = self.vocab.CLS_ID
        text_ids[1:seq_len-1]  = temp_ids
        text_ids[seq_len-1] = self.vocab.SEP_ID
        
        text_mask[:seq_len] = 1
        return torch.LongTensor(text_ids), torch.LongTensor(text_mask), rate

    def data_collate(self, datalist):
        texts, text_masks, rates = zip(*datalist)
        texts = torch.stack(texts)
        text_masks = torch.stack(text_masks)
        rates = torch.LongTensor(rates)
        return texts, text_masks, rates

    def __iter__(self):
        datapath = numpy.random.choice(self.dataloc, 1)[0]
        data_collect = []
        with gzip.open(datapath, "rb") as f:
            for l in f:
                try:
                    data = self.jsonconvert(l)
                except:
                    continue
                data_collect.append(data)
                if len(data_collect) >= self.batch_size:
                    yield self.data_collate(data_collect)
                    data_collect = []


class Sentiment_Full(nn.Module):
    def __init__(self, bert_model_name):
        super().__init__()
        self.encoder = Sentence_Encoder(bert_model_name)
        self.sentiment = Sentiment(self.encoder.hidden_size)
    
    def forward(self, sentence, attention_mask, labels=None):
        _, pooled_output = self.encoder(sentence, attention_mask=attention_mask)
        return self.sentiment(pooled_output, labels)


def train(epochs, learning_rate, bert_model_name, saved_model, dataloader):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    
    model = Sentiment_Full(bert_model_name).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    
    if os.path.exists(saved_model):
        checkpoints = torch.load(saved_model, map_location={'cuda:0':str(device)})
        model.load_state_dict(checkpoints)
    
    for epoch in range(epochs):
        for it, data in enumerate(dataloader):
            input_ids = data[0].to(device)
            mask = data[1].to(device)
            labels = data[2].to(device)
            loss = model(input_ids, mask, labels)
            loss.backward()
        
            optimizer.step()
            optimizer.zero_grad()
            print('{}/{} loss {}'.format(epoch, it, loss))

            if it>0 and it%1000 == 0:
                torch.save(model.state_dict(), saved_model)
                
        torch.save(model.state_dict(), saved_model)



if __name__ == "__main__":
    batch_size = 200
    max_seq_len = 200
    epochs = 100
    learning_rate = 0.001
    bert_model_name = "/home/pzhu/data/bert/bert-base-uncased"
    data_loc = "/home/pzhu/data/amazon-reviews-pds/"
    saved_model = '/home/pzhu/data/amazon-reviews-pds/sentiment.pt'

    data = DataReader(data_loc, bert_model_name=bert_model_name, max_seq_len=max_seq_len, batch_size=batch_size)
    train(epochs = epochs, learning_rate=learning_rate, bert_model_name=bert_model_name, saved_model=saved_model, dataloader=data)
    

