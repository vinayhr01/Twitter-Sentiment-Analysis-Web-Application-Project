import re
from nltk.stem.wordnet import WordNetLemmatizer 
import itertools
import numpy as np
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string


class sentiment_analysis_code:

    def cleaning(self, text):
        text = text.lower()
        text = re.sub(r'(\\u[0-9A-Fa-f]+)', ' ', text)       
        text = re.sub(r'[^\x00-\x7f]',r' ',text)
        text = re.sub('@[^\s]+',' ',text)
        text = " ".join(re.split(r'\s|-', text))
        text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))',' ',text)
        text = re.sub(r'#covid', r' covid ', text)
        text = re.sub(r'#corona', r' corona ', text)
        text = re.sub(r'#coronavirus', r' coronavirus ', text)
        text = re.sub(r'#([^\s]+)', r' ', text)
        text = ''.join([i for i in text if not i.isdigit()])
        text = re.sub(r"(\!)\1+", ' ', text)
        text = re.sub(r"(\?)\1+", ' ', text)
        text = re.sub(r"(\.)\1+", ' ', text)
        text = re.sub(':\)|;\)|:-\)|\(-:|:-D|=D|:P|xD|X-p|\^\^|:-|\^\.\^|\^\-\^|\^\_\^|\,-\)|\)-:|:\'\(|:\(|:-\(|:\S|T\.T|\.\_\.|:<|:-\S|:-<|\\-\*|:O|=O|=\-O|O\.o|XO|O\_O|:-\@|=/|:/|X\-\(|>\.<|>=\(|D:', ' ', text)
        text = re.sub('&amp;', 'and', text)
        
        #stoplist = stopwords.words('english')
        stop_words = set(stopwords.words('english'))
        stop_words.discard('not') 
        stop_words.discard('and')
        stop_words.discard('but')

        translator = str.maketrans('', '', string.punctuation)
        text = text.translate(translator) # Technique 7: remove punctuation

        tokens = nltk.word_tokenize(text)
        tokens = [tokens for tokens in tokens if not tokens in stop_words]

        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]

        tagged = nltk.pos_tag(tokens) # Technique 13: part of speech tagging 
        allowedWordTypes = ["J","R","V","N"] #  J is Adjective, R is Adverb, V is Verb, N is Noun. These are used for POS Tagging
        final_text = []
        for w in tagged:

            if (w[1][0] in allowedWordTypes):
                final_word = sentiment_analysis_code().addCapTag(w[0])
                final_word = lemmatizer.lemmatize(final_word)
                final_text.append(final_word)
                text = " ".join(final_text) 
        
        return text
    
        
    def addCapTag(self, word):
        """ Finds a word with at least 3 characters capitalized and adds the tag ALL_CAPS_ """
        if(len(re.findall("[A-Z]{3,}", word))):
            word = word.replace('\\', '' )
            transformed = re.sub("[A-Z]{3,}", "ALL_CAPS_"+word, word)
            return transformed
        else:
            return word

    def get_tweet_sentiment(self, tweet):
        #cleaning of tweet
        sid_obj = SentimentIntensityAnalyzer()
        sentiment_dict = sid_obj.polarity_scores(sentiment_analysis_code().cleaning(tweet))

        if sentiment_dict['compound'] > 0 :
            return 'Positive'
        else :
            return 'Negative'