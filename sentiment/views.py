from django.shortcuts import render, redirect, HttpResponse
from django.contrib import messages
from .forms import Sentiment_Typed_Tweet_analyse_form
from .sentiment_analysis_code import sentiment_analysis_code
from .forms import Sentiment_Imported_Tweet_analyse_form
from .tweepy_sentiment import Import_tweet_sentiment
import csv
import pandas as pd
import matplotlib.pyplot as plotter
import numpy as np
import matplotlib.pyplot as plt ;plt.rcdefaults()
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
import numpy as np
import pandas as pd
# plotting
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score
from collections import Counter


# sklearn
from sklearn.svm import LinearSVC #Linear SVM
from sklearn.naive_bayes import ComplementNB #Naive Bayes
from sklearn.linear_model import LogisticRegression #Logistic Regression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report
    
def sentiment_analysis(request):
    return render(request, 'home/sentiment.html')

def sentiment_analysis_type(request):
    if request.method == 'POST':
        form = Sentiment_Typed_Tweet_analyse_form(request.POST)
        analyse = sentiment_analysis_code()
        
        if form.is_valid():
            tweet = form.cleaned_data['sentiment_typed_tweet']
            sentiment = analyse.get_tweet_sentiment(tweet)
            args = {'tweet':tweet, 'sentiment':sentiment}
            return render(request, 'home/sentiment_type_result.html', args)

    else:
        form = Sentiment_Typed_Tweet_analyse_form()
        return render(request, 'home/sentiment_type.html')

def sentiment_analysis_import(request):
    if request.method == 'POST':
        form = Sentiment_Imported_Tweet_analyse_form(request.POST)
        tweet_text = Import_tweet_sentiment()
        analyse = sentiment_analysis_code()

        if form.is_valid():
            handle = form.cleaned_data['sentiment_imported_tweet']
            dict={}
            a=[]

            
            if handle in ["#COVID19", "#CovidIsNotOver", "#CovidVaccines", "#Coronavirus", "#CoronavirusUpdates", "#Corona"]:
                    list_of_tweets = tweet_text.get_hashtag(handle)
                    list_of_tweets_and_sentiments = []
                    for i in list_of_tweets:
                        list_of_tweets_and_sentiments.append(['@' + i[0], i[1], i[2], i[3]])
                        a.append(['@' + i[0], i[1], i[2], i[3]])
                    
                    df = pd.DataFrame(a, columns=['Tweet Username', 'Tweet Text', 'Cleaned Tweet Text', 'Result'])   
                    df.to_csv('compare.csv', index=False) 
                    
                    df = pd.read_csv("compare.csv")
                    comment_words = ''

                    # iterate through the csv file
                    for val in df['Cleaned Tweet Text']:
                            
                            # typecaste each val to string
                            val = str(val)

                            # split the value
                            tokens = val.split()
                            
                            comment_words += val

                    wordcloud = WordCloud(width = 800, height = 800,
                                                    background_color ='white',
                                                    min_font_size = 10).generate(comment_words)

                    # plot the WordCloud image				
                    plt.figure(figsize = (8, 8), facecolor = None)
                    plt.imshow(wordcloud)
                    plt.axis("off")
                    plt.tight_layout(pad = 0)
                    cloudfile = wordcloud.to_file("sentiment/static/home/images/cloud1.png")
                    
                    data = pd.read_csv("finalD.csv", encoding_errors='ignore', low_memory=False)
                    data = data.dropna()
                    data = data.reset_index(drop=True)
                    XX=data['Cleaned Tweet Text']
                    YY=data['Result']

                    XX_train, XX_test, YY_train, YY_test = train_test_split(XX,YY,test_size=0.2,random_state=42)
                    tfacc = TfidfVectorizer(ngram_range=(3,3))
                    xx_train = tfacc.fit_transform(XX_train) 
                    xx_test = tfacc.transform(XX_test)

                    print("Xtrain =",len(XX_train))
                    print("Xtest =",len(XX_test))
                    print("ytrain =",len(YY_train))
                    print("ytest =",len(YY_test))
                    count2=len(XX_train)

                    BNBmodel = ComplementNB()
                    BNBmodel.fit(xx_train, YY_train)
                    y_predBNB = BNBmodel.predict(xx_test)
                    
                    score = round(accuracy_score(YY_test, y_predBNB)*100, 3)
                    pscore = round(precision_score(YY_test, y_predBNB, average="micro")*100, 3)
                    print("Accuracy of Naive Bayes:   %0.3f" % score)
                    print("Precision of Naive Bayes:   %0.3f" % pscore)
                    
                    LGmodel = LogisticRegression(C=1, max_iter=1000)
                    LGmodel.fit(xx_train, YY_train)
                    y_predLG = LGmodel.predict(xx_test)
                    
                    score2 = round(accuracy_score(YY_test, y_predLG)*100, 3)
                    pscore2 = round(precision_score(YY_test, y_predLG, average="micro")*100, 3)
                    print("Accuracy of Logistic Regression:  %0.3f" % score2)
                    print("Precision of Logistic Regression:  %0.3f" % pscore2)
                    
                    SVMaccModel = LinearSVC()
                    SVMaccModel.fit(xx_train, YY_train)
                    y_predaccSVM = SVMaccModel.predict(xx_test)
                    score1 = round(accuracy_score(YY_test, y_predaccSVM)*100, 3)
                    pscore1 = round(precision_score(YY_test, y_predaccSVM, average="micro")*100, 3)
                    print("Accuracy of SVM: %0.3f" % score1)
                    print("Precision of SVM: %0.3f" % pscore1)

                    
                    tfidf_vectorizer = TfidfVectorizer()
                    tfidf_xtrain = tfidf_vectorizer.fit_transform(XX_train) 
                    tfidf_xtest = tfidf_vectorizer.transform(XX_test)
                    
                    SVCmodel = LinearSVC()
                    SVCmodel.fit(tfidf_xtrain, YY_train)
                    y_predSVC = SVCmodel.predict(tfidf_xtest)
                    

                    ml_pred = pd.read_csv("compare.csv")
                    ml_pred1 = tfidf_vectorizer.transform(ml_pred['Cleaned Tweet Text'])
                    test_pred = SVCmodel.predict(ml_pred1) 
                    
                    dicti = {'Prediction': test_pred}
                    df1 = pd.DataFrame(next(iter(dicti.values())), columns=['ML Result'])
                    out = pd.merge(ml_pred, df1, left_index=True, right_index=True)
                    out.to_csv("compareres.csv", index=False)
                    ou = pd.read_csv('compareres.csv', encoding_errors='ignore')
                    
                    ou = ou[ou['ML Result'] != 'Neutral']

                    count=len(ou)
                    poscont=len(ou[ou['Result']=='Positive'])
                    negcont=len(ou[ou['Result']=='Negative'])
                    
                    labels = ['Positive','Negative']
                    result2=open('results/resVader.csv', 'w')
                    result2.write("Parameter,Value" + "\n")
                    result2.write("Positive Tweets" + "," +str(poscont) + "\n")
                    result2.write("Negative Tweets" + "," +str(negcont) + "\n")
                    result2.close()
                    
                    df =  pd.read_csv('results/resVader.csv')
                    acc = df["Value"]
                    alc = df["Parameter"]
                    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
                    explode = (0.1, 0, 0)  
                    
                    fig = plt.figure()
                    plt.bar(alc, acc,color=colors)
                    plt.xlabel('Parameter')
                    plt.ylabel('Value')
                    plt.title('Sentiment analysis by Vader on realtime tweets')
                    fig.savefig('sentiment/static/home/images/resVader.png') 

                    count1=len(ou)
                    print("Count of tweets from ML", len(ou))
                    poscont1=len(ou[ou['ML Result']=='Positive'])
                    print("Positive tweets counts from ML =", len(ou[ou['ML Result']=='Positive']))
                    negcont1=len(ou[ou['ML Result']=='Negative'])
                    print("Negative tweets counts from ML =", len(ou[ou['ML Result']=='Negative']))
                    
                    print("Count of tweets from Vader", len(ou))
                    print("Positive tweets counts from Vader =", len(ou[ou['Result']=='Positive']))
                    print("Negative tweets counts from Vader =", len(ou[ou['Result']=='Negative']))
                    
                    labels = ['Positive','Negative']
                    result2=open('results/resML.csv', 'w')
                    result2.write("Parameter,Value" + "\n")
                    result2.write("Positive Tweets" + "," +str(poscont1) + "\n")
                    result2.write("Negative Tweets" + "," +str(negcont1) + "\n")
                    result2.close()
                    
                    df =  pd.read_csv('results/resML.csv')
                    acc = df["Value"]
                    alc = df["Parameter"]
                    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
                    explode = (0.1, 0, 0)  
                    
                    fig = plt.figure()
                    plt.bar(alc, acc,color=colors)
                    plt.xlabel('Parameter')
                    plt.ylabel('Value')
                    plt.title('Sentiment analysis by ML on realtime tweets')
                    fig.savefig('sentiment/static/home/images/resML.png') 


                    labels = ['Naive Bayes', 'Logistic Regression', 'SVM']
                    result2=open('results/resAcc.csv', 'w')
                    result2.write("Parameter,Value" + "\n")
                    result2.write("Naive Bayes" + "," +str(score) + "\n")
                    result2.write("Logistic Regression"+ "," + str(score2) + "\n")
                    result2.write("SVM" + "," +str(score1) + "\n")
                    result2.close()
                    
                    df =  pd.read_csv('results/resAcc.csv')
                    acc = df["Value"]
                    alc = df["Parameter"]
                    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
                    explode = (0.1, 0, 0)  

                    fig = plt.figure()
                    plt.bar(alc, acc,color=colors)
                    plt.xlabel('Parameter')
                    plt.ylabel('Value')
                    plt.title('Accuracy Comparsion')
                    fig.savefig('sentiment/static/home/images/resAcc.png')
                    
                    vad_ml_sentiment = ou.values.tolist()

                    res = ''
                    for i in ou['Cleaned Tweet Text']:
                        res += i
                    res_split = res.split()
                    cnt = Counter(res_split)
                    most = cnt.most_common(200)

                    most = [list(i) for i in most]
                    
                    r = ''
                    for i in most:
                        r += i[0] + ' '
                    
                    trans = tfidf_vectorizer.transform([r])
                    prediction = SVCmodel.predict(trans)
                    se = prediction[0]
                    
                    args = {'vad_ml_sentiment':vad_ml_sentiment, 'handle':handle, 'overall_senti': se, 'most':most, 'Total':count1, 'Positive_Count': poscont1, 'Negative_Count':negcont1, 'Positive_Count1': poscont, 'Negative_Count1': negcont, 'Naive':score, 'Logistic': score2, 'SVM': score1, 'total': count2}
                    return render(request, 'home/sentiment_import_result.html', args)
                
            messages.error(request, "Please give an appropriate hashtag.")
            return render(request, 'home/sentiment_import.html')

    else:
        form  = Sentiment_Imported_Tweet_analyse_form()
        return render(request, 'home/sentiment_import.html')