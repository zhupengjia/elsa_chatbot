#!/usr/bin/env python
import sys, h5py, re, os, numpy
sys.path.append("..")
from chatbot_end2end.reader.reader_base import ReaderBase

outdata = "data.h5"
new_name = "cornell"
if os.path.exists(outdata):
    os.remove(outdata)
h5file = h5py.File(outdata, 'w')
h5datasets = {}

for f in sys.argv[1:]:
    with h5py.File(f, 'r') as hf:
        keys = hf.keys()
        for k in keys:
            if "response_mask" in k:
                name = re.split("_", k)[-1]
                break
        for k in keys:
            if name in k:
                new_k = re.sub(name, new_name, k)
            else:
                new_k = k
            
            if not new_k in h5datasets:
                kshape = hf[k].shape 
                initshape = list(kshape)
                initshape[0] = 0
                chunkshape = list(kshape)
                chunkshape[0] = 1
                maxshape = list(kshape)
                maxshape[0] = None
    
                h5datasets[new_k] = h5file.create_dataset(k, tuple(initshape),
                                                      dtype=hf[k].dtype,
                                                      chunks=tuple(chunkshape),
                                                      compression='lzf',
                                                      maxshape=tuple(maxshape))
            
            print("merging {}, {}".format(f, new_k))
            ReaderBase.add_trace(h5datasets[new_k], hf[k].value)
