# Sentiment Analysis of Twitter Data on Economic Crisis due to COVID-19

## Introduction:

In this project, we analyze the twitter posts or tweets that are tweeted about COVID-19 Economic impact on different countries by different twitter users.

Sentiment analysis, also referred to as opinion mining, is an approach to natural language processing (NLP) that identifies the sentiment as either **Positive or Negative** behind a body of text . This is a popular way for organizations to determine and categorize opinions about a product, service, opinion or idea.

## Procedure:

We have a trained three different machine learning models Naive Bayes, Support Vector Machine (SVM) and Logistic Regression on finalD.csv file and accuracies are compared.

The project provided SVM with highest accuracy and it has been used to classify new tweets fetched in real-time from twitter based on economic crisis.

The VADER library is a popular 3rd party Social media Sentiment Analyzer which is also used in this project for prediction on same new tweets fetched real-time.

## Steps to run this project:

After forking this repository,
Execute below command in terminal for packages

```
pip install -r requirements.txt
```

Then, run python in a new terminal, and execute the below commands

```
import nltk
nltk.download('punkt')
```

```
import nltk
nltk.download('wordnet')
```

```
import nltk
nltk.download('averaged_perceptron_tagger')
```

Run the project as

```
python manage.py runserver
```

## Conclusion:

The results classified by ML model SVM and VADER are found to be comparable.
