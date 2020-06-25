"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os

# Data dependencies
import pandas as pd

#Inspecting
import numpy as np 
import pandas as pd 
from time import time
import re
import string
import os
import emoji
from pprint import pprint
import collections

#visualisation
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from PIL import Image
import plotly.express as px


# Balance data
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample

#Cleaning
import nltk
from nltk import PorterStemmer
from nltk.probability import FreqDist

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")
my_dataset2 = 'resources/test.csv'
data2 = pd.read_csv(my_dataset2)

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Prediction", "Information",'eda']

	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page
	if selection == "Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with ML Models")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as: {}".format(prediction))
	if selection == 'eda':
		st.image("https://raw.githubusercontent.com/Dominic-byte/classification-predict-streamlit-template/developing/test/resources/dataexploration.png",
                  use_column_width= True)
		# EDA
		my_dataset = 'resources/train.csv'
		st.info('View Original Data Set')
		# To Improve speed and cache data
		@st.cache(persist=True,allow_output_mutation=True)
		def explore_data(dataset):
			df = pd.read_csv(os.path.join(dataset))
			return df 

		# Our Dataset
		data = explore_data(my_dataset)

		# Show raw Dataset
		if st.checkbox("Preview DataFrame"):
			
			if st.button("Head"):
				st.write(data.head())
			if st.button("Tail"):
				st.write(data.tail())
			else:
				st.write(data.head(2))

		# Show Entire Dataframe
		if st.checkbox("Show All DataFrame"):
			st.dataframe(data)

		#Define Dataframe for pie chart plot
		df_pie = data.groupby('sentiment').count().reset_index()
		df_pie['sentiment'].replace([-1,0,1,2],['negative Sentiment = -1','neutral Sentiment = 0','positve Sentiment = 1','News Sentiment = 2'],inplace =True)

		#Show distribution of target variable
		st.info('View Distribution of Sentiment')
		if st.checkbox('Preview Plot'):
			if st.button('Bar Plot'):
				fig1 = sns.factorplot('sentiment',data = data, kind='count',size=6,aspect = 1.5, palette = 'PuBuGn_d')
				st.markdown("<h1 style='text-align: center; color: black;'>Distribution of Sentiment</h1>", unsafe_allow_html=True)
				st.pyplot(fig1)	
			if st.button('Pie Chart'):
				fig2 = px.pie(df_pie, values='message', names='sentiment',color_discrete_map={'negative Sentiment = -1':'lightcyan','neutral Sentiment = 0':'cyan','positve Sentiment = 1':'royalblue','News Sentiment = 2':'darkblue'})
				st.markdown("<h1 style='text-align: center; color: black;'>Climate Sentiment Pie Chart</h1>", unsafe_allow_html=True)
				st.plotly_chart(fig2)
		
		#Cleaning of text before tokenisation, stemming and removal of stop words
		def data_preprocessing(train,test):
			def remove_capital_words(df,column):
				df_Lower = df[column].map(lambda x: x.lower())
				return df_Lower
			train['tidy_tweet'] = remove_capital_words(train,'message')
			test['tidy_tweet'] = remove_capital_words(test,'message')
			contra_map = {
							"ain't": "am not ",
							"aren't": "are not ",
							"can't": "cannot",
							"can't've": "cannot have",
							"'cause": "because",
							"could've": "could have",
							"couldn't": "could not",
							"couldn't've": "could not have",
							"didn't": "did not",
							"doesn't": "does not",
							"don't": "do not",
							"hadn't": "had not",
							"hadn't've": "had not have",
							"hasn't": "has not",
							"haven't": "have not",
							"he'd": "he would",
							"he'd've": "he would have",
							"he'll": "he will",
							"he'll've": "he will have",
							"he's": "he is",
							"how'd": "how did",
							"how'd'y": "how do you",
							"how'll": "how will",
							"how's": "how is",
							"i'd": "I would",
							"i'd've": "I would have",
							"i'll": "I will",
							"i'll've": "I will have",
							"i'm": "I am",
							"i've": "I have",
							"isn't": "is not",
							"it'd": "it would",
							"it'd've": "it would have",
							"it'll": "it will",
							"it'll've": "it will have",
							"it's": "it is",
							"let's": "let us",
							"ma'am": "madam",
							"mayn't": "may not",
							"might've": "might have",
							"mightn't": "might not",
							"mightn't've": "might not have",
							"must've": "must have",
							"mustn't": "must not",
							"mustn't've": "must not have",
							"needn't": "need not",
							"needn't've": "need not have",
							"o'clock": "of the clock",
							"oughtn't": "ought not",
							"oughtn't've": "ought not have",
							"shan't": "shall not",
							"sha'n't": "shall not",
							"shan't've": "shall not have",
							"she'd": "she would",
							"she'd've": "she would have",
							"she'll": "she will",
							"she'll've": "she will have",
							"she's": "she is",
							"should've": "should have",
							"shouldn't": "should not",
							"shouldn't've": "should not have",
							"so've": "so have",
							"so's": "so is",
							"that'd": "that would",
							"that'd've": "that would have",
							"that's": "that is",
							"there'd": "there would",
							"there'd've": "there would have",
							"there's": "there is",
							"they'd": "they would",
							"they'd've": "they would have",
							"they'll": "they will",
							"they'll've": "they will have",
							"they're": "they are",
							"they've": "they have",
							"to've": "to have",
							"wasn't": "was not",
							"we'd": "we would",
							"we'd've": "we would have",
							"we'll": "we will",
							"we'll've": "we will have",
							"we're": "we are",
							"we've": "we have",
							"weren't": "were not",
							"what'll": "what will",
							"what'll've": "what will have",
							"what're": "what are",
							"what's": "what is",
							"what've": "what have",
							"when's": "when is",
							"when've": "when have",
							"where'd": "where did",
							"where's": "where is",
							"where've": "where have",
							"who'll": "who will",
							"who'll've": "who will have",
							"who's": "who is",
							"who've": "who have",
							"why's": "why is",
							"why've": "why have",
							"will've": "will have",
							"won't": "will not",
							"won't've": "will not have",
							"would've": "would have",
							"wouldn't": "would not",
							"wouldn't've": "would not have",
							"y'all": "you all",
							"y'all'd": "you all would",
							"y'all'd've": "you all would have",
							"y'all're": "you all are",
							"y'all've": "you all have",
							"you'd": "you would",
							"you'd've": "you would have",
							"you'll": "you will",
							"you'll've": "you will have",
							"you're": "you are",
							"you've": "you have"}
			contractions_re = re.compile('(%s)' % '|'.join(contra_map.keys()))
			def contradictions(s, contractions_dict=contra_map):
				def replace(match):
					return contractions_dict[match.group(0)]
				return contractions_re.sub(replace, s)
			train['tidy_tweet']=train['tidy_tweet'].apply(lambda x:contradictions(x))
			test['tidy_tweet']=test['tidy_tweet'].apply(lambda x:contradictions(x))
			def replace_url(df,column):
				df_url = df[column].str.replace(r'http.?://[^\s]+[\s]?', 'urlweb ')
				return df_url
			train['tidy_tweet'] = replace_url(train,'tidy_tweet')
			test['tidy_tweet'] = replace_url(test,'tidy_tweet')
			def replace_emoji(df,column):
				df_emoji = df[column].apply(lambda x: emoji.demojize(x)).apply(lambda x: re.sub(r':[a-z_&]+:','emoji ',x))
				return df_emoji
			train['tidy_tweet'] = replace_emoji(train,'tidy_tweet')
			test['tidy_tweet'] = replace_emoji(test,'tidy_tweet')
			def remove_digits(df,column):
				df_digits = df[column].apply(lambda x: re.sub(r'\d','',x))
				return df_digits
			train['tidy_tweet'] = remove_digits(train,'tidy_tweet')
			test['tidy_tweet'] = remove_digits(test,'tidy_tweet')	
			def remove_patterns(df,column):
				df_char = df[column].apply(lambda x:  re.sub(r'[^a-z# ]', '', x))
				return df_char
			train['tidy_tweet'] = remove_patterns(train,'tidy_tweet')
			test['tidy_tweet'] = remove_patterns(test,'tidy_tweet')   							
			return train,test
		(train_set,test_set) = data_preprocessing(data,data2)

		def tok_stemm_stopwords_transform(train,test):
			train['token'] = train['tidy_tweet'].apply(lambda x: x.split())
			test['token'] = test['tidy_tweet'].apply(lambda x: x.split())
			stemmer = PorterStemmer()
			train['stemming'] = train['token'].apply(lambda x: [stemmer.stem(i) for i in x]) # stemming
			test['stemming'] = test['token'].apply(lambda x: [stemmer.stem(i) for i in x]) # stemming
			#create my own stop words from analysis and comparing with general stopwords
			stopwords_own =[ 'i','me','my','myself','we','our','ours','ourselves','you','your','yours','yourself','yourselves','he','him',
                'his','himself','she','her','hers','herself','it','itself','they','them','their','theirs','themselves','what',
                'which','who','whom','this','that','these','those','am','is','are','was','were','be','been','being','have','has',
                'had','having','do','does','did','doing','a','an','the','and','but','if','or','because','as','until','while',
                'of','at','by','for','with','about','against','between','into','through','during','before','after','above',
                'below','to','from','up','down','in','out','on','off','over','under','again','further','then','once','here',
                'there','when','where','why','how','all','any','both','each','few','more','most','other','some','such','only',
                'own','same','so','than','too','very','s','t','can','will','just','should','now','d','ll','m','o','re','ve','y',
               #my own stopwords found from analysis
                'u','doe','going','ha','wa','l', 'thi','becaus','rt']		
			# def remove_strop_words(df,column):
			def remove_stopwords(df,column):
				df_stopwords = df[column].apply(lambda x: [item for item in x if item not in stopwords_own])
				return df_stopwords
			train['stem_no_stopwords'] = remove_stopwords(train,'stemming')
			test['stem_no_stopwords'] = remove_stopwords(test,'stemming')
			#Transformation
			def convert_st_str(df,column):
				df_str = df[column].apply(lambda x: ' '.join(x))
				return df_str
			train['clean_tweet'] = convert_st_str(train,'stem_no_stopwords')
			test['clean_tweet'] = convert_st_str(test,'stem_no_stopwords')
			return train,test
		(clean_train_df,clean_test_df) = tok_stemm_stopwords_transform(train_set,test_set)

		#Sentiment of 2
		news_words =' '.join([text for text in clean_train_df['clean_tweet'][clean_train_df['sentiment'] == 2]])
		
		# Create and generate a word cloud image:
		wordcloud = WordCloud(width=2000, height=1500, random_state=21, max_font_size=200).generate(news_words)

		# Display the generated image:
		plt.imshow(wordcloud, interpolation='bilinear')
		plt.axis("off")
		plt.show()
		st.pyplot()


# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
