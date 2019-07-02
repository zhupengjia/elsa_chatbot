FROM python:3-slim
MAINTAINER Pengjia Zhu <zhupengjia@gmail.com>

EXPOSE 5000/tcp

ENV WORKDIR /opt/chatbot
ENV BUILD_VERSION b1.0

RUN apt-get update &&\
    apt-get install -y --no-install-recommends libhdf5-dev gcc gfortran&&\
    rm -rf /var/lib/apt/lists/*

# python packages
RUN pip3 install --no-cache --upgrade pip setuptools wheel flask sleekxmpp python-telegram-bot&&\
    pip3 install --no-cache pyyaml bidict tqdm h5py numpy scipy pandas sklearn spacy nltk xlrd&&\
    pip3 install --no-cache https://download.pytorch.org/whl/cpu/torch-1.1.0-cp37-cp37m-linux_x86_64.whl &&\
    python3 -m spacy download en &&\
    python3 -c "import nltk;nltk.download('vader_lexicon')"

RUN pip3 install --no-cache https://f002.backblazeb2.com/file/pzhu-public/pytorch_pretrained_bert-0.6.2-py3-none-any.whl &&\
    pip3 install --no-cache https://f002.backblazeb2.com/file/pzhu-public/nlptools-0.5.1-py2.py3-none-any.whl

WORKDIR $WORKDIR

ENV BOT_VERSION a0.1

ADD chatbot_end2end chatbot_end2end

ADD interact.py interact.py

CMD python interact.py -c ${BOT_CONFIG}
#CMD tail -f /dev/null
