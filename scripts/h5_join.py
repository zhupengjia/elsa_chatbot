#!/usr/bin/env python
import sys, h5py, re, os, numpy
sys.path.append("..")
from chatbot_end2end.reader.reader_base import ReaderBase

outdata = "data.h5"
new_name = "cornell"
if os.path.exists(outdata):
    os.remove(outdata)
h5file = h5py.File("data.h5", 'w')
h5datasets = {}

all_names = []
all_data = {}
for f in sys.argv[1:]:
    if os.path.exists(f):
        print("merging ", f)
        with h5py.File(f, 'r') as hf:
            keys = hf.keys()
            for k in keys:
                if "response_mask" in k:
                    name = re.split("_", k)[-1]
                    all_names.append(name)
                    break
            for k in keys:
                if name in k:
                    new_k = re.sub(name, new_name, k)
                else:new_k = k
                if not new_k in all_data:
                    all_data[new_k] = []
                all_data[new_k].append(hf[k].value)

for k in all_data.keys():
    data = numpy.concatenate(all_data[k], axis=0)
    kshape = data.shape
    
    initshape = list(kshape)
    initshape[0] = 0
    chunkshape = list(kshape)
    chunkshape[0] = 1
    maxshape = list(kshape)
    maxshape[0] = None
    
    print(k, kshape, tuple(initshape), data.dtype, tuple(chunkshape), tuple(maxshape))

    h5datasets[k] = h5file.create_dataset(k, tuple(initshape), dtype=data.dtype,
                                          chunks=tuple(chunkshape), compression='lzf',
                                          maxshape=tuple(maxshape))
    
    ReaderBase.add_trace(h5datasets[k], data)

