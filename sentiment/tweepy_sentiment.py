from tweepy import OAuthHandler
from tweepy import API
from tweepy import Cursor
from datetime import datetime, date, time, timedelta
from collections import Counter
import sys
import tweepy
import numpy as np
import pandas as pd
import snscrape.modules.twitter as sntwitter
import sentiment.sentiment_analysis_code as senti


class Import_tweet_sentiment:
        consumer_key="QIqgjITOfksfMW4lRLDacQ"
        consumer_secret="R8x0xN9iSKXGNxUtGKA2hgnlIhh5INZIOdgEfxzk"
        access_token="1401204486-BeLUAuruh294KeJX8NXvdqjCeZOQcLl6HWmMlgA"
        access_token_secret="pwjiLF42TbORaXtkCS5Oc24qywOU0eFN0esVcibA"
        def tweet_to_data_frame(self, tweets):
                df = pd.DataFrame(data=[tweet.text for tweet in tweets], columns=['Tweets'])
                return df

        def get_hashtag(self, hashtag):
                auth = OAuthHandler(self.consumer_key, self.consumer_secret)
                auth.set_access_token(self.access_token, self.access_token_secret)
                #q = "(covid AND economic crisis OR income OR tax OR gdp OR corona OR pandemic OR covid crisis OR poor OR #COVID19 OR #Corona OR #Coronavirus) -filter:retweets"
                
                #q = "lang:en(((India OR Sri Lanka OR Pakistan) AND (covid AND economic crisis)) AND (income OR tax OR gdp OR corona OR pandemic OR covid crisis OR poor OR #COVID19 OR #Corona OR #Coronavirus))"# since:2019-12-05 until:2022-12-05)" #for snscrape
                
                q = "((India AND Pakistan) AND (economic crisis OR financial OR income OR tax OR gdp OR #COVID19 OR #Corona OR pay OR loan OR inflation OR market OR gst)) -filter:retweets" # for tweepy
                
                # Create API object
                api = tweepy.API(auth)

                atweets = []
                
                for tweet in tweepy.Cursor(api.search, q, lang="en").items(700):
                        cleaned_tweet = senti.sentiment_analysis_code.cleaning(self, tweet.text)
                        tweet_sentiment = senti.sentiment_analysis_code.get_tweet_sentiment(self, cleaned_tweet)
                        atweets.append([tweet.user.screen_name, tweet.text, cleaned_tweet , tweet_sentiment])

                #print(atweets)
                
                return atweets
                
                """ atweets = []
                for i, tweet in enumerate(sntwitter.TwitterSearchScraper(q).get_items()):
                        if i >= 10:
                                break
                        else:
                                cleaned_tweet = senti.sentiment_analysis_code.cleaning(self,tweet.rawContent)
                                tweet_sentiment = senti.sentiment_analysis_code.get_tweet_sentiment(self, cleaned_tweet)
                                atweets.append([tweet.user.username, tweet.rawContent, cleaned_tweet , tweet_sentiment])
                return atweets """