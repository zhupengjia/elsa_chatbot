FROM alpine:latest
MAINTAINER Pengjia Zhu <zhupengjia@gmail.com>

EXPOSE 5000/tcp

ENV WORKDIR /opt/chatbot
ENV BUILD_VERSION b1.0

RUN apk update
RUN apk add --no-cache python3 git wget bash nano python3-dev gcc musl-dev g++ cython3 lapack-dev
RUN apk add --no-cache make gfortran

# HDF5
RUN wget https://s3.amazonaws.com/hdf-wordpress-1/wp-content/uploads/manual/HDF5/HDF5_1_10_5/source/hdf5-1.10.5.tar.bz2 \
        && tar xf hdf5-1.10.5.tar.bz2 \
        && cd hdf5-1.10.5 \
        && ./configure --prefix=/usr --enable-cxx --with-zlib=/usr/include,/usr/lib/x86_64-linux-gnu \
        && make -j4 \
        && make install \
        && cd .. \
        && rm -rf hdf5-1.10.5 \
        && rm -rf hdf5-1.10.5.tar.bz2 \
        && export HDF5_DIR=/usr

RUN HDF5_LIBDIR=/usr/lib HDF5_INCDIR=/usr/include python3 -m pip --no-cache-dir install \
 --no-binary=h5py h5py

# python packages
RUN pip3 install --no-cache --upgrade pip setuptools wheel
RUN pip3 install --no-cache pyyaml bidict tqdm h5py
RUN pip3 install --no-cache numpy scipy
RUN pip3 install --no-cache h5py pandas sklearn spacy nltk
RUN pip3 install --no-cache https://download.pytorch.org/whl/cpu/torch-1.1.0-cp37-cp37m-linux_x86_64.whl
RUN pip3 install --no-cache https://f002.backblazeb2.com/file/pzhu-public/nlptools-0.5.0-py2.py3-none-any.whl
RUN pip3 install --no-cache git+https://github.com/facebookresearch/fastText.git
RUN pip3 install --no-cache git+https://github.com/bung87/whatlangid.git
RUN pip3 install --no-cache git+https://github.com/huggingface/pytorch-pretrained-BERT.git
RUN python3 -m spacy download en
RUN python3 -c "import nltk;nltk.download('vader_lexicon')"

WORKDIR $WORKDIR

ENV BOT_VERSION a0.1

ADD chatbot_end2end chatbot_end2end


