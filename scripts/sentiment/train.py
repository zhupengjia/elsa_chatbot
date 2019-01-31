#!/usr/bin/env python
import glob, numpy, gzip, os, json, torch, sys, h5py
sys.path.append("../..")
import torch.nn as nn
from chatbot_end2end.model.sentence_encoder import Sentence_Encoder
from chatbot_end2end.model.sentiment import Sentiment
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim


class DataReader(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data = h5py.File(data_path, 'r')
   
    def __len__(self):
        return self.data["ids"].shape[0]

    def __getitem__(self, i):
        return torch.LongTensor(self.data["ids"][i]), torch.LongTensor(self.data["masks"][i]), torch.tensor(self.data["label"][i], dtype=torch.long)


class Sentiment_Full(nn.Module):
    def __init__(self, bert_model_name):
        super().__init__()
        self.encoder = Sentence_Encoder(bert_model_name)
        self.sentiment = Sentiment(self.encoder.hidden_size)
    
    def forward(self, sentence, attention_mask, labels=None):
        _, pooled_output = self.encoder(sentence, attention_mask=attention_mask)
        return self.sentiment(pooled_output, labels)


def train():
    batch_size = 400
    epochs = 100
    learning_rate = 0.001
    num_workers = 1
    bert_model_name = "/home/pzhu/data/bert/bert-base-uncased"
    saved_model = '/home/pzhu/data/amazon-reviews-pds/sentiment.pt'
    data_path = "/home/pzhu/data/amazon-reviews-pds/reviews.h5"
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    
    dataset = DataReader(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

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
    train()
    

