#!/usr/bin/env python
# coding: utf-8

# # Table of Contents
# ---
# 1. [Introduction](#intro)
# 
#   * Problem Statement
#   * Sentiments Description
# ---
# 2. [Imports](#data)
#   * Comet
#   * Installations
#   * Packages 
#   * Data
# ---
# 3. [Explonatory Data Analysis](#EDA)
# 
# 3.1. Missing values and blanks 
#   
# 3.2. Data Summary statistics
#   
# 3.3. Sentiment Visual Distributions
#    * Percentages of each Sentiments 
#    * Tweets per sentiment 
#    * Number of words per tweet
#    * Word Cloud 
#    * Mentions
#    * Hashtags 
#    * Url Counts
#    * Retweets
#    * Name Entities 
# ---
# 4. [Data Preprocessing](#data)
#   * Remove Contractions
#   * Remove Non-english tweets 
#   * Clean tweets 
#   * Lemmatization
#   * Converting HTML Entities
# ---
# 5. [Analysis of cleaned Data]()
#   * Word Cloud 
#   * NER
# ---
# 6. [Feature Engineering]()
#   * Classification Models 
#   * Validation split
#   * Pipeline
# ---
# 7. [modeling]()
#   * Training the model 
#   * F1 score Results 
#   * Testing on cleaned data
# ---
# 8. [Resampling]()
#   * Smote
#   * Downsampling 
#   * Testing on resampled Data
# ---
# 9. [Hyperparameter Tuning]()
#   * Linear SVC
# ---
# 10. [Final Model Testing]()
# ---
# 11. [Submission]()
# ---
# 12. [Conclusion]()
# ---
# 13. [Acknowlegement]()
# ---

# # 1. Introduction

# ### Climate change project description 
# 
# Many companies offer products and services that are environmentally friendly and sustainable, in line with their values and ideals. They would like to determine how people perceive climate change and whether or not they believe it is a real threat. Twitter has become one of the most cost-effective marketing strategy platforms used by companies to engage their target markets.
# This notebook describes the process to classifying tweets by sentiment using Natural Language Processing techniques.
# It describes the initial data exploration, as well as implementation of different machine model classifiers used for predictions .
# 
# This will be done by importing necesarry libraries as well as the training and test datasets. Data cleaning follows together with exploratory data analysis.We then wrap up the notebook by diving into different classification techniques under the Modelling section which will be followed by insights and a conclusion.
# 
# Classification is a process of categorizing a given set of data into classes, It can be performed on both structured or unstructured data. The process starts with predicting the class of given data points. The classes are often referred to as target, label or categories.

# ### Problem statement
# Build a robust Machine Learning Model that will be able to predict a person’s belief in Climate Change based on their Tweet Data, allowing companies to gain access into customer sentiment

#   ### Sentiment Description
#   
#         Class   Sentiments    Description
#          2	   News:         the tweet link to factual news about climate change
#          1       Pro:          the tweet supports the belief of man-made climate change
#          0	   Neutral:      the tweet neither supports nor refutes the belief of man-made climate change
#         -1       Anti:         the tweet does not believe in man-made climate change

#  # 2. Imports

# ### a Comet 
# 
# * Install Comet
# * Import Experiment from Comet
# * Create an experiment instance

# In[1]:


pip install comet_ml


# In[2]:


from comet_ml import Experiment


# ### b Packages 

# In[5]:


pip install langdetect


# In[6]:


pip install contractions


# In[7]:


pip install fasttext


# In[8]:


pip install spacy


# In[9]:


pip install imblearn


# In[10]:


import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from matplotlib.colors import ListedColormap
get_ipython().run_line_magic('matplotlib', 'inline')

# imports for Natural Language  Processing
import re
import os
import nltk
import string
import time
import fasttext
import spacy.cli
from langdetect import detect
import contractions
import unicodedata
import numpy as np
import pandas as pd
import xgboost
from sklearn import metrics
from nltk.corpus import stopwords
from html.parser import HTMLParser
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('stopwords')
nltk.download('punkt')

# Classification Models

from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier

# Performance Evaluation
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from scikitplot.metrics import plot_roc, plot_confusion_matrix
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report, confusion_matrix

# Import library for train test split
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV

#Resampling techniques
from collections import Counter 
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

# Ignore warnings
import warnings
warnings.simplefilter(action='ignore')

#spacy
spacy.cli.download('en_core_web_sm')


# ---
# After importing the dependencies, both the train data and test data are read in. From train dataset and test dataset, pandas methods are used to examine summary statistics such as:
# 
# That the correct number of rows and columns have been read in
# Account for missing values and correct if there are any missing values
# The data types that make up the datasets
# The columns that make up the datasets
# Pull out tweets from a column to see the typical body/structure of the tweets
# 
# ---
# 
# ### Data
# 

# Train.csv: Dataset that contains all the variables that should be used to train the model
# 
# Test.csv : Dataset that contains variables that will be used to test the model

# In[11]:


train = pd.read_csv('C:\Users\hp\Downloads\edsa-climate-change-belief-analysis-2022.csv')

test = pd.read_csv('"C:\Users\hp\Downloads\edsa-climate-change-belief-analysis-2022.csv')


# ### Variables definitions
# 
# Sentiment: Sentiment of tweet
# 
# Message: Tweet body
# 
# Tweetid: Twitter unique id

# # 3. Exploratory Data Analysis 
# 
# Exploratory data analysis is an approach to analyzing data sets to summarize their main characteristics, often with visual methods. primarily EDA is for seeing what the data can tell us beyond the formal modeling or hypothesis testing task.This approach for data analysis uses many tools(mainly graphical to maximize insight into a data set, extract important variables, detect outliers and anomalies, amongst other details that is missed when looking at DataFrame.
# 
# This step is very important especially when we model the data in order to apply Machine Learning techniques.

# ### 3.1 Missing values and blank strings 

# In[12]:


train.isnull().sum()


# In[13]:


test.isnull().sum()


# In[14]:


blanks = []  

for ms in train.itertuples():  
    if type(ms)==str:            
        if ms.isspace():         
            blanks.append(i)     
        
print(len(blanks), 'blanks: ', blanks)


# In[15]:


blanks = []  

for ms in test.itertuples():  
    if type(ms)==str:            
        if ms.isspace():         
            blanks.append(i)     
        
print(len(blanks), 'blanks: ', blanks)


# **Observation:**
# * The training dataset has no null values or any blank tweet rows within it.
# * The test data also has no missing values or blank tweet rows

# ### 3.2 Data summary statisitics

# In[16]:


train.head()


# In[17]:


train.info()


# In[18]:


test.head()


# In[19]:


test.info()


# **Observations:**
# * The train and test datasets contains one categorical column called 'message'
# * The train dataset contains three columns
# * The test data contains two columns, it excluses the predictive (y) column sentiment 
# * Some tweets contain twitter handles (e.g @RawStory), numbers (e.g year 2016), hashtags (e.g #TodayinMaker# WIRED) and re-tweets (RT).
# * Some tweets contain names of ogarnisations, continents and countries.
# * New lines are represented by '\n' in the tweet string.

# ### 3.3 Sentiment visual distributions

# In[20]:


#make a copy of the train dataset

train1 = train.copy()


# ####  A. The percentage of each sentiment row
# 
# The following code will count the number of rows each sentiment has and the total percentage it carries 

# #### B. Tweets per Sentiment 

# In[21]:


'''
We calculate the number of tweets per Sentiment,

we then plot the class distributions results'''

# Number of tweets per sentiment
class_distribution = pd.DataFrame(list(train1['sentiment'].value_counts()),
                          index=['Pro', 'News', 'Neutral', 'Anti'],
                          columns=['Count'])
sns.set()
sns.barplot(x=class_distribution.index, y=class_distribution.Count, 
           palette="Blues_d")
plt.title('Class Distributions')


# **Observations:**
# * The sum of the tweets relating to news,neutral and anti is less than half of the total tweets.
# * Looking at the distribution we are able to see that the data is imbalanced, most tweets are skewed to the Pro sentiment category supporting the belief of man-made climate change.

# #### C. Number of words per tweet

# In[22]:


#identify the row we want 
tweet = train['message']


# In[23]:


# creating a new DataFrame
tweet_df = pd.DataFrame(tweet)

# Add sentiment column to the tweets dataframe
tweet_df['sentiment'] = train1['sentiment']

tweet_df.head()


# In[24]:


'''
We creating a Collection of written text of each sentiment class '''

news_tweets = ' '.join([text for text in tweet_df['message']
                        [tweet_df['sentiment'] == 2]])
pro_tweets = ' '.join([text for text in tweet_df['message']
                       [tweet_df['sentiment'] == 1]])
neutral_tweets = ' '.join([text for text in tweet_df['message']
                           [tweet_df['sentiment'] == 0]])
anti_tweets = ' '.join([text for text in tweet_df['message']
                        [tweet_df['sentiment'] == -1]])


# In[25]:


# Visualising sentiment class 
full_title = ['Popular words for News tweets',
              'Popular words for Pro tweets',
              'Popular words for Neutral tweets',
              'Popular words for Anti tweets']
#creating a list for the visuals 
tweet_list = [news_tweets, pro_tweets,
              neutral_tweets, anti_tweets]

plt.rcParams['figure.figsize'] = [40, 5]

for i, sent in enumerate(tweet_list):
    plt.subplot(1, 4, i + 1)
    freq_dist = nltk.FreqDist(sent.split(' '))
    df = pd.DataFrame({'Word': list(freq_dist.keys()),
                      'Count' : list(freq_dist.values())})

    df = df.nlargest(columns='Count', n=15)

    ax = sns.barplot(data=df, y='Word', x='Count')
    plt.title(full_title[i])
    plt.show()


# * The graphs above showcase the evidence of noise. A lot of stop words are picked up as being important which include: (the, to, and, also of). Also, in the graph labeled popular for news tweets, there is a punctuation (a dash -) picked up as an important word.

# #### D. Wordcloud
# 
# Creating a word cloud to visualizate tweet keywords and text data.
# This is to highlight popular or trending terms based on frequency of use and prominence.
# The larger the word in the visual the more common the word is on tweet messages.

# In[26]:


# Create word clouds of the most common words in each sentiment class
wc = WordCloud(width=600, height=400, 
               background_color='black', colormap='Dark2',
               max_font_size=150, random_state=42)

plt.rcParams['figure.figsize'] = [20, 15]

# Create subplots 
for i in range(0, len(tweet_list)):
    wc.generate(tweet_list[1])
    
    plt.subplot(2, 2, i + 1)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.title(full_title[i])
    
plt.show()


# **Observation:**
# * The size of the word indicates the relevance in the tweet.
# * The most popular words in all four classes are climate change, global warming and belief.
# * The pro and anti groups include a number of words that might be expected in each group.
# * In the word clouds there is evidence of noisy text which include words such as https, webside, co and RT. These do not assist us in our classification, rather they add noise, we will have another look at it when the noise have been removed.
# * https occurs frequently in pro climate change tweets, implying that many links are being shared around the topic of climate change. These could be links to petitions, websites and/or articles related to climate change. Interesting to note: https only occurs in the top 25 words for the pro climate change class. Why aren't we seeing more links in the news class?

# #### e. Mention Analysis
# 
# A mention is a Tweet that contains another person's username anywhere in the body of the Tweet. We collect these messages, as well as all the replies. Including multiple usernames in a Tweet, all of those people you mentioned will see your Tweet.

# In[27]:


"""
We calculate the number of mentions we have for each sentiment in a tweet
"""

# mention count 
train1['mentions'] = train1['message'].apply(lambda x: len([i for i in str(x) if i == '@']))

#plot the number of mentions
plt.subplot(1,3,3)
sns.stripplot(y='mentions', x='sentiment', data=train1, jitter=True)
plt.title('Number of mentions')
plt.ylabel('')
plt.xlabel('')
fig = plt.gcf()
fig.set_size_inches( 23, 5)

plt.show()


#  **Observation:**
# 
# The pro and neutral setiments seem to have the most mentions in the tweets compared to the news and anti sentiment classes. 

# #### f. Extracting hashtags
# 
# People use the hashtag symbol (#) before a relevant keyword or phrase in their Tweet to categorize those Tweets and help them show more easily in Twitter search. Clicking or tapping on a hashtagged word in any message shows you other Tweets that include that hashtag. Hashtags can be included anywhere in a Tweet

# In[28]:


# Creating a function to extract hashtags from tweets

def extract_hashtags(x):
    hashtags = []
    for i in x:
        ht = re.findall(r'#(\w+)', i)
        hashtags.append(ht)
        
    return hashtags


# In[29]:


# Extracting hashtags from tweets
news_h = extract_hashtags(tweet_df['message']
                              [tweet_df['sentiment'] == 2])
pro_h = extract_hashtags(tweet_df['message']
                          [tweet_df['sentiment'] == 1])
neutral_h = extract_hashtags(tweet_df['message']
                              [tweet_df['sentiment'] == 0])
anti_h = extract_hashtags(tweet_df['message']
                          [tweet_df['sentiment'] == -1])

# hashtag list
hashtags = [sum(news_h, []), sum(pro_h, []),
            sum(neutral_h, []),sum(anti_h, [])]

# Visualising the Hashtags
ft = [' Hashtags on the News sentiment',
              ' Hashtags on the Pro sentiment',
              ' Hashtags on the Neutral sentiment',
              ' Hashtags on the Anti sentiment']

plt.rcParams['figure.figsize'] = [50, 5]

for i, sent in enumerate(hashtags):
    plt.subplot(1, 4, i + 1)
    freq_dist = nltk.FreqDist(sent)
    df = pd.DataFrame({'Hashtag': list(freq_dist.keys()),
                      'Count' : list(freq_dist.values())})

    df = df.nlargest(columns='Count', n=15)

    ax = sns.barplot(data=df, y='Hashtag', x='Count')
    plt.title(ft[i])
    plt.show()


# **Observation:**
# 
# * We can see that the top 5 hashtags have similar words like Climate, climate change, Trump and Before the flood
# * Before the flood is a popular hashtags used in pro climate change tweets, this refers to a 2016 documentary where actor Leonardo DiCaprio meets with scientists, activists and world leaders to discuss the dangers of climate change and possible solutions.
# * In the anti climate change tweets MAGA (Make America great again) is the top popular hashtag. It is a slogan that was often used by Donald Trump during his campaign for elections in 2016. This soon became a trending hashtag to use to show support for Donald Trump., 

# #### g. Url counts 

# In[30]:


# extracting urls
train1['urls'] = train1['message'].apply(lambda x: len([i for i in x.lower().split() if 'http' in i or 'https' in i]))

# ploting the number of urls
plt.subplot(1,3,3)
sns.stripplot(y='urls', x='sentiment', data=train1, jitter=True)
plt.title('Number of urls')
plt.ylabel('')
plt.xlabel('')
fig = plt.gcf()
fig.set_size_inches( 23, 5)

plt.show()


# **Observation:**There is not much difference between the number of urls in each setiment. 

# ### I. Retweets
# 
# Twitter allows a user to retweet, or RT another users tweets. We see RT as a popular word in the above visuals. This is great for creating trends, but not useful for sentiment analysis. Now we will remove the duplicates to get a clearer picture of our data set.

# In[31]:


# Class distribution for set of retweeted-tweets and set without retweets
plt.figure(figsize = (8,5))
train1['retweet'] = train1['message'].apply(lambda tweet: 1 if tweet.startswith('RT @') else 0)
sns.countplot(x='retweet', data=train1, palette='dark', hue='sentiment')
plt.title('Number of Retweets Per Sentiment Class',fontsize=14)
plt.xlabel('Retweet')
plt.ylabel('Count')
plt.legend(title='Class')
plt.show()


# **Observations:**
# 
# The Pro sentiment class seems to have more tweets retweeted with over 5000 retweets. while other sentiment classes have less than 2000 retweets. looks like evryone is retweeting positive climate change tweets more than others.

# ### J. Finding entities 

# In[32]:


nlp = spacy.load('en_core_web_sm')

def entities(df):
    df_index = 0

    for tweet in df['message']:
        tweet = nlp(tweet)

    for entity in tweet.ents:
        df.loc[df_index, 'message'] = df.loc[df_index, 'message'].replace(str(entity.text), str(entity.label_))
        df_index += 1

        return df


# In[33]:


entities(train)


# ### k. Twitter Handles
# 
# A Twitter handle is the username that appears at the end of your unique Twitter URL. Twitter handles appear after the @ sign in your profile URL and it must be unique to your account. A Twitter name, on the other hand, is simply there to help people find the company they're looking for.

# In[34]:


# Creating a function to extract handles from tweets
def extract_handles(x):
    handles = []
    for i in x:
        h = re.findall(r'@(\w+)', i)
        handles.append(h)
        
    return handles


# In[35]:


# Extracting handles from tweets
news_h = extract_handles(tweet_df['message']
                              [tweet_df['sentiment'] == 2])
pro_h = extract_handles(tweet_df['message']
                          [tweet_df['sentiment'] == 1])
neutral_h = extract_handles(tweet_df['message']
                              [tweet_df['sentiment'] == 0])
anti_h = extract_handles(tweet_df['message']
                          [tweet_df['sentiment'] == -1])

# handle lists 
handles = [sum(news_h, []), sum(pro_h, []), sum(neutral_h, []),
           sum(anti_h, [])]

# Visualising the Handles
full_title = ['Twitter Handles on the News sentiment',
              'Twitter Handles on the Pro sentiment',
              'Twitter Handles on the Neutral sentiment',
              'Twitter Handles on the Anti sentiment']

plt.rcParams['figure.figsize'] = [50, 5]

for i, sent in enumerate(handles):
    plt.subplot(1, 4, i + 1)
    freq_dist = nltk.FreqDist(sent)
    df = pd.DataFrame({'Handle': list(freq_dist.keys()),
                      'Count' : list(freq_dist.values())})

    df = df.nlargest(columns='Count', n=15)

    ax = sns.barplot(data=df, y='Handle', x='Count')
    plt.title(full_title[i])
    plt.show()


# **Observations**
# 
# From the Visuals above we can all that ...
# * The most popular News handles are actual news broadcaster accounts
# * The most popular Pro handles seem to be celebrity accounts & news accounts.
# * Trump features most for most popular Anti & Neutral tweets.

# # 4. Data preprocessing

# In[36]:


df = train.copy() #make copy of Train DataFrame
df1 = test.copy() #make copy of test Dataframe


# In[37]:


df.head()


# ### a. Removing Contractions
# 
# Contractions are words or combinations of words which are shortened by dropping letters and replacing them by an apostrophe.
# 
# For examples: we’re = we are; we’ve = we have; I’d = I would
# 
# In NLP we have to deal with contrctiobs because:
# 
# A computer does not know that contractions are abbreviations for a sequence of words. Therefore, a computer considers we’re and we are to be two completely different things and does not recognize that these two terms have the exact same meaning.
# 
# Contractions increase the dimensionality of the document-term matrix
# 
# Therefore contractions will be removed from the message column in the DataFrame

# In[38]:


#Removing the contractions for both the train and Train DataFrames

#Remove contractions on Train Dataset
df['message'] = df['message'].apply(lambda x: [contractions.fix(word) for word in x.split()])
df['message'] = [' '.join(map(str, l)) for l in df['message']]

#Remove Contractions on Test Dataset
df1['message'] = df1['message'].apply(lambda x: [contractions.fix(word) for word in x.split()])
df1['message'] = [' '.join(map(str, l)) for l in df1['message']]

df.head()


# ### b. Remove non-english tweets
# 
# When models are trained, they are often unable to distiguish a language change. 

# In[39]:


def detect_language(tweet):
    """
    This function detects the different lanuages written in the tweets,
    to make it easy to remove the other laguages 
    """
    return detect(tweet)

# Language Detection
df['language'] = df['message'].apply(detect_language)


# In[40]:


# Creating a new dataframe that will show the language type and how many tweets are in that lanugauage 
lang= df['language'].value_counts()
lang_df = pd.DataFrame({'ISO Code':lang.index, 'Rows':lang.values})
lang_df.set_index('ISO Code', inplace=True)
lang_df.head() #Showcasing the first 5


# We are able to notice that in our train data there is non english strings. This includes id which is Bahasa Indonesia, it which is Italiano and fr which is français, langue française. The question is do we drop the non-engish words or do we translate them? Or is it better to leave it as it?
# 
# According to 'toward Data Science' translation is not a good option for sentiment analysis, it causes a drop in accuracy of 16%. For sentiment analysis stop words and word embeddings are useful. 
# 
# So we tested out the different of f1 scores if we were to leave the non-english tweets as compared to dropping them in the test dataset

# In[41]:


mydicta = {'Logistic Regression': ['0.70542', '0.71194'],
          'Support Vector Classifier': ['0.71728', '0.72072'],
          'Linear SVC': ['0.72083', '0.72107'],
          'XGBoost': ['0.67804', '0.68847']
         }
p = pd.DataFrame(mydicta.items(), columns=['Model', 'F-1 Score'])
p


# To our suprise, the four models we did this test on all performed better when we did not drop the non-english coloumns. This could because it meant that the model had more data to train on. Therefore, we concluded with not dropping any columns.

# ### c. Cleaning Tweets
# 
# The DataFrame will be further cleaned to remove noise, this includes entities such as mentions, url's, hashtags, numbers, punctuation, emojis/characters, ascii's and whitespaces.

# In[42]:


def Cleantweet (tweet):
    """
    This function will remove the noises from the DataFrame which include removing: mentions, urls,
    hashtags, numbers, punctuations, emojis/characters, acsii's and white spaces. 
    
    Before the function removes noise entities, it will convert the strings into lowercase
    
    Input is datatype 'str': tweet (noisy tweet)
    
    Output is datatype 'str': tweet (cleaned tweet)
    """
        
    #convert to lowercase
    tweet = tweet.lower()
    
    #remove mentions 
    tweet = re.sub('@[\w]*','',tweet)
    
    #remove urls
    tweet = re.split('https:\/\/.*', str(tweet))[0]
    
    #remove hashtags
    tweet = re.sub(r'#\w*','', tweet)
    
    #remove numbers 
    tweet = re.sub(r'\d+','', tweet)
    
    #remove punctuation
    tweet = re.sub(r"[,.;':@#?!\&/$]+\ *", ' ', tweet)
    
    #remove emojis 
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    tweet = regrex_pattern.sub(r'', tweet)
    
     #remove acsii
    tweet = unicodedata.normalize('NFKD', tweet).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    
    #remove extra whitespaces 
    tweet = re.sub(r'\s\s+', ' ', tweet)
    
    #remove space in front of tweet
    tweet = tweet.lstrip(' ')
    
    return tweet 


# In[43]:


#Pass the message column of the dataframe through the Cleantweet function 
df['message'] = df['message'].apply(Cleantweet)
df1['message'] = df1['message'].apply(Cleantweet)
df.head()


# ### Parts of speech tagging and lemmatization
# 
# Part-of-Speech tagging is a well-known task in Natural Language Processing. It refers to the process of classifying words into their parts of speech (also known as words classes or lexical categories). This is a supervised learning approach.

# The PoS of a word is important to properly obtain the word’s lemma, which is the canonical form of a word (this happens by removing time and grade variation, in English).
# 
# For example, what is the canonical form of “living”? “to live” or “living”? It depends semantically on the context and, syntactically, on the PoS of “living”. If “living” is an adjective (like in “living being” or “living room”), we have base form “living”. If it is a noun (“he does it for living”) it is also “living”. But if it is a verb (“he has been living here”), it is “lo live”. This is an example of a situation where PoS matters.
# 
# ![image.png](attachment:image.png)
# 
# Considering these uses, you would then use PoS Tagging when there’s a need to normalize text in a more intelligent manner (the above example would not be distinctly normalized using a Stemmer) or to extract information based on word PoS tag. 
# 
# Tiago Duque (2020): Building a Part of Speech Tagger

# Therefore, documents are going to use different forms of a word, such as organize, organizes, and organizing. 
# 
# The goal of lemmatization is to reduce inflectional forms and sometimes derivationally related forms of a word to a common base form. 
# 
# For instance:
# 
# The result of this mapping of text will be something like:
# the boy's cars are different colors $\Rightarrow$ the boy car be differ color
# 
# Lemmatization usually refers to doing things properly with the use of a vocabulary and morphological analysis of words, normally aiming to remove inflectional endings only and to return the base or dictionary form of a word, which is known as the lemma . If confronted with the token saw, stemming might return just s, whereas lemmatization would attempt to return either see or saw depending on whether the use of the token was as a verb or a noun. 

# In[44]:


def lemmatization(df):
    """
    This function will tokenized the message column, 
    then assign a part of speech tag before lemmatization
    
    Input is datatype dataframe
    
    Output is datatype dataframe
    """
    
    df['tokenized'] = df['message'].apply(word_tokenize)
    df['pos_tags'] = df['tokenized'].apply(nltk.tag.pos_tag)
    
    
    def get_wordnet_pos(tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN
        
    df['wordnet_pos'] = df['pos_tags'].apply(lambda x: [(word, get_wordnet_pos(pos_tag)) for (word, pos_tag) in x])
    
    lemmatizer = WordNetLemmatizer()
    df['lemmatized'] = df['wordnet_pos'].apply(lambda x: [lemmatizer.lemmatize(word, tag) for word, tag in x])
    df['lemmatized'] = [' '.join(map(str, l)) for l in df['lemmatized']]
    return df
    


# In[45]:


# Lemmatise tweets
df = lemmatization(df)
df1 = lemmatization(df1)
df.head()


# # Converting (HTML) entities:
# An HTMLParser instance is fed HTML data and calls handler methods when start tags, end tags, text, comments, and other markup elements are encountered. Removal of words in our data like '&amp', '&lt' (which are basically used in HTML).

# In[46]:


html_parser = HTMLParser()

df['messages'] = df['lemmatized'].apply(lambda x: html_parser.unescape(x))
df.head()


# # 5. Analysis of data after cleaning data

# ### Word cloud 

# In[47]:


''' We are ploting the wordcloud of a cleaned Data 

    where we removed Noise.'''

# Plot wordcloud for Pro Class
wordcloud = WordCloud(background_color='black', width=800, height=400).generate(' '.join(df[df['sentiment'] == 1]
                                          ['messages']))
plt.figure( figsize=(12,6))
plt.imshow(wordcloud)
plt.axis("off")
plt.title('Pro')
plt.show()

# Plot for News class
wordcloud = WordCloud(background_color='black', width=800, height=400).generate(' '.join(df[df['sentiment'] == 2]
                                          ['messages']))
plt.figure( figsize=(12,6))
plt.imshow(wordcloud)
plt.axis("off")
plt.title('News')
plt.show()

# Plot for Neutral class 
wordcloud = WordCloud(background_color='black', width=800, height=400).generate(' '.join(df[df['sentiment'] == 0]
                                          ['messages']))
plt.figure( figsize=(12,6))
plt.imshow(wordcloud)
plt.axis("off")
plt.title('Neutral')
plt.show()

#Plot for Anti class 
wordcloud = WordCloud(background_color='black', width=800, height=400).generate(' '.join(df[df['sentiment'] == -1]
                                          ['messages']))
plt.figure( figsize=(12,6))
plt.imshow(wordcloud)
plt.axis("off")
plt.title('Anti')
plt.show()


# **Observations** 
# 
# we have now removed the Noise and we are able to see the most common words used in all sentiments being 
# * Climate change
# * Global Warming
# * Trump
# 

# # 6. Feature Engineering
# 
# A pipeline will be used to build classification models. 
# 
# Classification models: 
# * Logistic Regression
# * Multinomial Naive Bayes
# * Random Forest Classifier
# * Support Vector Classifier
# * Linear SVC
# * K Nearest Neighbours Classifier
# * Decision Tree Classifier
# * AdaBoost Classifier
# * SGD Classifier
# * XGBoost

# ### Validation split
# The cleaned train data will be split into feautures and target variables. Then the data will be split into a train and validation set. The purpose of spliting the data into a train and validation set is to be able to evaluate the performance of the models. The evaluation will help choose the best model for submission. 

# In[48]:


# Create a copy of the cleaned dataframe with selected columns
f = df[['sentiment', 'lemmatized', 'tweetid']].copy()
f.head()


# In[49]:


# Seperate features and target variables
X = f['lemmatized']
y = f['sentiment']

# Create train and validation set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# ### Pipeline
# 
# In the pipeline, first it will vectorize the text data, the fit the model. 
# 
# Our feauture variables are still classified as text data (strings). Unforutnatly, machines cannot understand raw text as they only can interpret numerical data. Therefore it is important to vectorize the data. 
# 
# > The TfidfVectorizer will tokenize documents, learn the vocabulary and inverse document frequency weightings, and allow you to encode new documents (Jason Brownlee, 2017)
# 
# The advatage of the TfidfVectorizer is that the resulting vectors are already scaled. The assigned word frequency scores by the TfidfVectorizer are normalized to values between 0 and 1 and the encoded document vectors can then be used directly with most machine learning algorithms

# In[50]:


#Logistic Regression
lr = Pipeline([('tfidf', TfidfVectorizer(sublinear_tf=True, 
                                         smooth_idf = True, 
                                         max_df = 0.3,
                                         ngram_range = (1, 2),
                                         stop_words='english')),
               ('clf', LogisticRegression(random_state=123, 
                                          multi_class='ovr',
                                          n_jobs=1, 
                                          C=1e5,
                                          max_iter = 4000))])
    
    
    
#Multinomial Naive Bayes
multi = Pipeline([('tfidf', TfidfVectorizer(sublinear_tf=True, 
                                            smooth_idf = True, 
                                            max_df = 0.3,
                                            ngram_range = (1, 2),
                                            stop_words='english')),
                  ('clf', MultinomialNB())])
    
    
    
#Random Forest Classifier
rf = Pipeline([('tfidf', TfidfVectorizer(sublinear_tf=True, 
                                         smooth_idf = True, 
                                         max_df = 0.3,
                                         ngram_range = (1, 2),
                                         stop_words='english')),
               ('clf', RandomForestClassifier(n_estimators=100, 
                                              max_depth=2, 
                                              random_state=0, 
                                              class_weight="balanced"))])
    
    
#Support Vector Classifier
svc = Pipeline([('tfidf', TfidfVectorizer(sublinear_tf=True, 
                                          smooth_idf = True, 
                                          max_df = 0.3,
                                          ngram_range = (1, 2),
                                          stop_words='english')),
                ('clf', SVC(gamma = 0.8, 
                            C = 10, 
                            random_state=42))])
    
    
    
#Linear SVC
linsvc = Pipeline([('tfidf', TfidfVectorizer(sublinear_tf=True, 
                                             smooth_idf = True, 
                                             max_df = 0.3,
                                             ngram_range = (1, 2),
                                             stop_words='english')),
                   ('clf', LinearSVC())])
    
    
    
#K Nearest Neighbours Classifier
kn = Pipeline([('tfidf', TfidfVectorizer(sublinear_tf=True, 
                                         smooth_idf = True, 
                                         max_df = 0.3,
                                         ngram_range = (1, 2),
                                         stop_words='english')),
               ('clf', KNeighborsClassifier(n_neighbors=3))])
    
    
    
#Decision Tree Classifier
dt = Pipeline([('tfidf', TfidfVectorizer(sublinear_tf=True, 
                                         smooth_idf = True, 
                                         max_df = 0.3,
                                         ngram_range = (1, 2),
                                         stop_words='english')),
               ('clf', DecisionTreeClassifier(random_state=42))])
    
    
    
#AdaBoost Classifier
ad = Pipeline([('tfidf', TfidfVectorizer(sublinear_tf=True, 
                                         smooth_idf = True, 
                                         max_df = 0.3,
                                         ngram_range = (1, 2),
                                         stop_words='english')),
               ('clf', AdaBoostClassifier(random_state=42))])
    
    
    
#SGD Classifier
sgd = Pipeline([('tfidf', TfidfVectorizer(sublinear_tf=True, 
                                          smooth_idf = True, 
                                          max_df = 0.3,
                                          ngram_range = (1, 2),
                                          stop_words='english')),
                ('clf', SGDClassifier(loss='hinge', 
                                      penalty='l2',
                                      alpha=1e-3, 
                                      random_state=42, 
                                      max_iter=5, 
                                      tol=None))])

    
#XGBoost
xg = Pipeline([('tfidf', TfidfVectorizer(sublinear_tf=True, 
                                         smooth_idf = True, 
                                         max_df = 0.3,
                                         ngram_range = (1, 2),
                                         stop_words='english')),
               ('clf', xgboost.XGBClassifier(learning_rate =0.1,
                                             n_estimators=1000,
                                             max_depth=5, 
                                             min_child_weight=1,
                                             gamma=0,
                                             subsample=0.8,
                                             colsample_bytree=0.8,
                                             nthread=4,
                                             seed=27))])


# # 7. Modeling and evaluation

# ### Train models

# In[51]:


# Logistic regression
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Multinomial Naive Bayes
multi.fit(X_train, y_train)
y_pred_multi = multi.predict(X_test)

# Random Forest Classifier
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Support Vector Classifier
svc.fit(X_train, y_train)
y_pred_svc = svc.predict(X_test)

# Linear SVC
linsvc.fit(X_train, y_train)
y_pred_linsvc = linsvc.predict(X_test)

# K Nearest Neighbours Classifier
kn.fit(X_train, y_train)
y_pred_kn = kn.predict(X_test)

# Decision Tree Classifier
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# AdaBoost Classifier
ad.fit(X_train, y_train)
y_pred_ad = ad.predict(X_test)

# SGD Classifier
sgd.fit(X_train, y_train)
y_pred_sgd = sgd.predict(X_test)

# XGBoost
xg.fit(X_train, y_train)
y_pred_xg = xg.predict(X_test)


# ## a. Logistic Regression
# 
# Logistic regression is a statistical model that makes use of a logistic function to model a binary dependent variable, however, multiclass classification with logistic regression can be done through the one-vs-rest scheme in which a separate model is trained for each class to predict whether an observation is that class or not (thus making it a binary classification problem)

# In[52]:


start = time.time()
lg_f1 = round(f1_score(y_test, y_pred_lr, average='weighted'),2)
print('Accuracy %s' % accuracy_score(y_pred_lr, y_test))
print("Model Runtime: %0.2f seconds"%((time.time() - start)))
report = classification_report(y_test, y_pred_lr, output_dict=True)
results = pd.DataFrame(report).transpose()
results


# In[53]:


plot_confusion_matrix(y_test, y_pred_lr, normalize=True,figsize=(8,8),cmap='winter_r')
plt.show()


# Logistic regression is able to successfully classify the tweets.
# This model classifies most tweets successfully with clear boundaries and less confusion surrounding the pro climate change class.
# The precision, accuracy and F1 scores have improved significantly for the pro, anti and neutral classes.
# There is a drop in the F1 score for the pro climate change class as the predictions become more balanced.
# The overall F1 score is 0.71 which is on target. Let's see if we can improve.

# ## b. Multinomial Naive Bayes
# 
# The Multinomial Naive Bayes model estimates the conditional probability of a particular feature given a class and uses a multinomial distribution for each of the features. The model assumes that each feature makes an independent and equal contribution to the outcome

# In[54]:


multi_f1 = round(f1_score(y_test, y_pred_multi, average='weighted'),2)
print('Accuracy %s' % accuracy_score(y_pred_multi, y_test))
print("Model Runtime: %0.2f seconds"%((time.time() - start)))
report = classification_report(y_test, y_pred_multi, output_dict=True)
results = pd.DataFrame(report).transpose()
results


# In[55]:


plot_confusion_matrix(y_test, y_pred_multi, normalize=True,figsize=(8,8),cmap='winter_r')
plt.show()


# Although the Naive Bayes model is a slight improvement on the random forest model it still performs poorly
# This model classifies most tweets as pro climate change with improved predictions for the news class.
# The precision, accuracy and F1 scores have improved significantly for the news class but remain low for neutral and anti.
# The overall F1 score is 0.64. Again this score could only be achieved since the majority of tweets are in fact pro climate change

# ## c. Random Forest Classifier
# 
# Random forest models are an example of an ensemble method that is built on decision trees (i.e. it relies on aggregating the results of an ensemble of decision trees). Decision tree machine learning models represent data by partitioning it into different sections based on questions asked of independent variables in the data. Training data is placed at the root node and is then partitioned into smaller subsets which form the 'branches' of the tree. In random forest models, the trees are randomized and the model returns the mean prediction of all the individual trees

# In[56]:


rf_f1 = round(f1_score(y_test, y_pred_rf, average='weighted'),2)
print('Accuracy %s' % accuracy_score(y_pred_rf, y_test))
print("Model Runtime: %0.2f seconds"%((time.time() - start)))
report = classification_report(y_test, y_pred_rf, output_dict=True)
pd.DataFrame(report).transpose()


# In[57]:


plot_confusion_matrix(y_test, y_pred_rf, normalize=True,figsize=(8,8),cmap='winter_r')
plt.show()


# From the confusion matrix above we notice that the random forest classification model does a very poor job on our data set. The model classifies all the tweets as pro climate change tweets.
# This results in precision, recall and F1 scores of zero for the anti, neutral and news classes.
# Tree based classification models are especially vulnerable to overfitting when the train data is imbalanced which is the case with our data. The model could be greatly improved by using resampling techniques such as oversampling the anti class and/or undersampling the pro class. This will allow the model to learn how to classify each class equally, improving its accuracy.
# The overall F1 score is 0.42. This is a relatively high score for a model that simply classifies all tweets into a single class. This score could only be achieved since the majority of the tweets are in fact pro climate change.

# ## d. Support Vector Classifier
# 
# A Support Vector Classifier is a discriminative classifier formally defined by a separating hyperplane. When labelled training data is passed to the model, also known as supervised learning, the algorithm outputs an optimal hyperplane which categorizes new data.

# In[58]:


svc_f1 = round(f1_score(y_test, y_pred_svc, average='weighted'),2)
print('Accuracy %s' % accuracy_score(y_pred_svc, y_test))
print("Model Runtime: %0.2f seconds"%((time.time() - start)))
report = classification_report(y_test, y_pred_svc, output_dict=True)
results = pd.DataFrame(report).transpose()
results


# In[59]:


plot_confusion_matrix(y_test, y_pred_svc, normalize=True,figsize=(8,8),cmap='winter_r')
plt.show()


# ## e. Linear SVC
# 
# The objective of a Linear Support Vector Classifier is to return a "best fit" hyperplane that categorises the data. It is similar to SVC with the kernel parameter set to ’linear’, but it is implemented in terms of liblinear rather than libsvm, so it has more flexibility in the choice of penalties and loss functions and can scale better to large numbers of samples.

# In[60]:


linsvc_f1 = round(f1_score(y_test, y_pred_linsvc, average='weighted'),2)
print('Accuracy %s' % accuracy_score(y_pred_linsvc, y_test))
print("Model Runtime: %0.2f seconds"%((time.time() - start)))
report = classification_report(y_test, y_pred_linsvc, output_dict=True)
results = pd.DataFrame(report).transpose()
results


# In[ ]:





# ## f. K Nearest Neighbours Classifier
# 
# The K Neighbours Classifier is a classifier that implements the k-nearest neighbours vote. In classification, the output is a class membership. An object is classified by a plurality vote of its neighbours, with the object being assigned to the class most common among its k-nearest neighbours.

# ## g. Decision Tree Classifier
# 
# Decision tree machine learning models represent data by partitioning it into different sections based on questions asked of independent variables in the data. Training data is placed at the root node and is then partitioned into smaller subsets which form the 'branches' of the tree.

# **Observations:**

# ## h. AdaBoost Classifier
# 
# The AdaBoost classifier is an iterative ensemble method that begins by fitting a classifier on the original dataset and then fits additional copies of the classifier on the same dataset. In the second step, the weights of incorrectly classified instances are adjusted such that subsequent classifiers focus more on difficult cases.

# In[66]:


ad_f1 = round(f1_score(y_test, y_pred_ad, average='weighted'),2)
print('accuracy %s' % accuracy_score(y_pred_ad, y_test))
print("Model Runtime: %0.2f seconds"%((time.time() - start)))
report = classification_report(y_test, y_pred_ad, output_dict=True)
pd.DataFrame(report).transpose()

