#!/usr/bin/env python3
import tweepy
import time
import pickle
import argparse
import zlib, os


parser = argparse.ArgumentParser(description='distinct-n')

parser.add_argument('-data', type=str, default="twitter_ids.txt",
                    help='Path to the *-train.pt file from preprocess.py')

opt = parser.parse_args()

# these arguments can be accessed in facebook developer.
consumer_key = sys.argv[1]
consumer_secret = sys.argv[2]
access_token = sys.argv[3]
access_token_secret = sys.argv[4]

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)


def zdump(value,filename):
    with open(filename,"wb",-1) as fpz:
        fpz.write(zlib.compress(pickle.dumps(value,-1),9))

def zload(filename):
    with open(filename,"rb") as fpz:
        value=fpz.read()
        try:return pickle.loads(zlib.decompress(value))
        except:return pickle.loads(value)


def main():
    twitter_ids_lines = open(opt.data, "r").readlines()
    dumpfile = "id2text.pkl"
    if os.path.exists(dumpfile):
        id2text = zload(dumpfile)
    else:
        id2text = {}

    utterance_ids = []
    for twitter_ids_line in twitter_ids_lines:
        ids = [int(_id) for _id in twitter_ids_line.strip().split("	")]
        ids = [i for i in ids if not i in id2text]
        utterance_ids += ids

    utterance_cnt = len(utterance_ids)
    for index in range(0, utterance_cnt, 100):
        id_list = utterance_ids[index: index+100]
        try:
            results = api.statuses_lookup(id_list)
            for result in results:
                id2text[result.id] = str(result.text).replace("\n", " ")
        except Exception as err:
            print(err)
            continue
        if index % 1000 == 0:
            zdump(id2text, dumpfile)
            print("Crawlling index = ", index)
            print("Percentage " + str(index * 1.0 / utterance_cnt) + " %")
            time.sleep(5)
        if index % 10000 == 0:
            time.sleep(120)
    zdump(id2text, dumpfile)


if __name__ == '__main__':
    main()
