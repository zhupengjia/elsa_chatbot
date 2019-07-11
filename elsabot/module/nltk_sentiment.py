#!/usr/bin/env python
from nltk.sentiment.vader import SentimentIntensityAnalyzer

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''


class NLTKSentiment:
    """
        Use NLTK for sentiment analyzer
    """
    def __init__(self):
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

    def __call__(self, sentence):
        """
            Input:
                - sentence: string

            Output:
                - float from -1 to 1
        """
        return self.sentiment_analyzer.polarity_scores(sentence)['compound']

