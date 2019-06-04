#!/usr/bin/env python3
import sys, pickle, zlib
def zdump(value,filename):
    with open(filename,"wb",-1) as fpz:
        fpz.write(zlib.compress(pickle.dumps(value,-1),9))

def zload(filename):
    with open(filename,"rb") as fpz:
        value=fpz.read()
        try:return pickle.loads(zlib.decompress(value))
        except:return pickle.loads(value)


d1 = zload(sys.argv[1])
print(len(d1))

for f in sys.argv[2:]:
    d2 = zload(f)
    print(len(d2))
    d1.update(d2)

print(len(d1))
zdump(d1, "1.pkl")



