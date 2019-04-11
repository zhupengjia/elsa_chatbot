#!/usr/bin/env python
import glob, os, gzip, shutil, json, time
import h5py, numpy
from nlptools.text  import Tokenizer

max_seq_len = 200
bert_model_name = "/home/pzhu/data/bert/bert-base-uncased"
data_folder = "/home/pzhu/data/amazon-reviews-pds/"
db_path = "/home/pzhu/data/amazon-reviews-pds/reviews.h5"

data_paths = glob.glob(os.path.join(data_folder, "*.json.gz"))

#num_lines = 0
#for d in data_paths:
#    with gzip.open(d, 'rb') as f:
#        num_lines += sum(1 for l in f)
#print(num_lines)
num_lines = 18186733

if os.path.exists(db_path):
    os.remove(db_path)

tokenizer = Tokenizer(tokenizer='bert', bert_model_name=bert_model_name)
vocab = tokenizer.vocab
def jsonconvert(jsonstr):
    data = json.loads(jsonstr)
    text = data['reviewText']
    rate = int(data["overall"]) - 1
    text_ids = numpy.zeros(max_seq_len)
    text_mask = numpy.zeros(max_seq_len)

    temp_ids = vocab(tokenizer(text))[:max_seq_len-2]
    seq_len = len(temp_ids) + 2

    text_ids[0] = vocab.BOS_ID
    text_ids[1:seq_len-1]  = temp_ids
    text_ids[seq_len-1] = vocab.EOS_ID
    
    text_mask[:seq_len] = 1
    return text_ids, text_mask, rate


def add_trace(dset, arr):
    dset.resize((dset.shape[0]+1, max_seq_len) )
    dset[-1,:] = arr
                

with h5py.File(db_path, "w") as h5file:
    id_set = h5file.create_dataset("ids", (0, max_seq_len), dtype='i', chunks=(1, max_seq_len), maxshape=(None, max_seq_len)) 
    mask_set = h5file.create_dataset("masks", (0, max_seq_len), dtype='i', chunks=(1, max_seq_len), maxshape=(None, max_seq_len)) 
    label_set = h5file.create_dataset("label", (num_lines,), dtype='i') 
    h5file.attrs["Nlines"] = num_lines
    ikey = 0
    for data_path in data_paths:
        with gzip.open(data_path, "rb") as f:
            for l in f:
                t1 = time.time()
                text_ids, text_mask, rate = jsonconvert(l)
                t2 = time.time()
                add_trace(id_set, text_ids)
                add_trace(mask_set, text_mask)
                
                label_set[ikey] = rate
                t3 = time.time()
                ikey += 1
                if ikey % 200 == 0:
                    print(data_path, ikey, t2-t1, t3-t2)

