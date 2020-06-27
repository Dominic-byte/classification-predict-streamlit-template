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
import altair as alt


# Balance data
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample

#Natural Language Toolkit
import nltk
from nltk import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
import spacy
sp = spacy.load('en_core_web_sm')

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file
from gensim.models import Word2Vec

# Load your raw data
raw = pd.read_csv("resources/train.csv")
#my_dataset2 = 'resources/test.csv'
#data2 = pd.read_csv(my_dataset2)

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

		#Lemmetization and Stemming
		st.subheader("**_Lemmetization and Stemming_**")
		st.info("Predict Lemmetization and  Stemming of your own words")
		# Creating a text box for user input
		tweet_text_ls = st.text_area("Enter Text","Type Here")

		#Lemmetization Predictor
		if st.button('Lemmetization'):
			text = sp(tweet_text_ls)
			pred_l = []
			for word in text:
				pred_l.append('Lemma for '+str(word)+' is '+str(word.lemma_))

			for p in pred_l:
				st.success("{}".format(p))

		#Stemming Predictor
		if st.button('Stemming'):
			stemmer = PorterStemmer()
			tokenizer = nltk.word_tokenize(tweet_text_ls)
			pred_l = []
			for token in tokenizer:
				pred_l.append('Stem for '+token+' is '+stemmer.stem(token))

			for p in pred_l:
				st.success("{}".format(p))	


		

		#Info
		st.subheader("**_Original Tweets_**")
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

		#Markdown explaining the distribtion of Target
		st.subheader("**_Distribution of Target_**")
		st.markdown('<p><ul><li>The positive sentiment counts are significantly higher followed by news, then neutral and lastly anti.', unsafe_allow_html=True)

		#Show distribution of target variable
		st.info('View Distribution of Sentiment')
		if st.button('Bar Plot'):
			@st.cache(persist=True,allow_output_mutation=True)
			def figure1(df):
				fig = sns.factorplot('sentiment',data = df, kind='count',size=6,aspect = 1.5, palette = 'PuBuGn_d')
				return fig
			fig1 = figure1(data)
			st.markdown("<h1 style='text-align: center; color: black;'>Distribution of Sentiment</h1>", unsafe_allow_html=True)
			st.pyplot(fig1)	
		if st.button('Pie Chart'):
			@st.cache(persist=True,allow_output_mutation=True)
			def figure2(df):
				fig = px.pie(df, values='message', names='sentiment',color_discrete_map={'negative Sentiment = -1':'lightcyan','neutral Sentiment = 0':'cyan','positve Sentiment = 1':'royalblue','News Sentiment = 2':'darkblue'})
				return fig
			fig2 = figure2(df_pie)
			st.markdown("<h1 style='text-align: center; color: black;'>Climate Sentiment Pie Chart</h1>", unsafe_allow_html=True)
			st.plotly_chart(fig2, use_container_width=True)

		#markdown to explain the clean data
		st.subheader("**_Clean Tweets_**")
		st.markdown("""
					<p><ul><li> Firslt, the cleaning of the data followed a process of using <a href="https://docs.python.org/3/howto/regex.html" target="_blank">Regex</a> to remove capital words, 
					replace urls, replace emojis, remove digits only keep certain characters within the text. For more information, 
					you may look at the following link <a href="https://towardsdatascience.com/sentiment-analysis-with-text-mining-13dd2b33de27l" target="_blank">Sentiment Analysis</a></li>
					<li> Secondly, the following methods were used to enable the natural language process library built 
					in python in order to clean the texts further. These methods were, <a href="https://www.nltk.org/api/nltk.tokenize.html" target="_blank">tokenization</a>,  <a href="https://pythonprogramming.net/stemming-nltk-tutorial/" target="_blank">stemming</a>
					and lastly removal of <a href="https://www.nltk.org/book/ch02.html" target="_blank">stopwords</a></li>
					<li>Finally, the cleaned tweets were transformed from a list (due to tokenization) to a string.</li></ul></p>
					""",unsafe_allow_html=True)
		
		#Cleaning of text before tokenisation, stemming and removal of stop words
		clean_train_df = pd.read_csv("resources/Clean_Train.csv")
		clean_test_df = pd.read_csv("resources/Clean_Test.csv")


		#Define Dataframe for more Analysis 
		EDA_df = clean_train_df[['sentiment','clean_tweet']]

		#Info
		st.info('View Clean Data Set')

		#View Clean Data
		@st.cache(persist=True)
		def explore_data_clean(df):
			df1 = df
			return df1

		# Our clean Dataset
		data_clean = explore_data_clean(EDA_df)

		# Show clean Dataset
		if st.checkbox("Preview showing clean DataFrame"):
			
			if st.button("Head of Clean Data"):
				st.write(data_clean.head())
			if st.button("Tail of Clean Data"):
				st.write(data_clean.tail())
			else:
				st.write(data_clean.head(2))

		# Show Entire Dataframe
		if st.checkbox("Show All  of Clean Dataframe"):
			st.dataframe(data_clean)

		#Preper Word2Vec
		@st.cache(persist=True,allow_output_mutation=True)
		def token(df):
			df1 = df['clean_tweet'].apply(lambda x: x.split()) #tokenising
			return df1
		tokenised_tweet = token(clean_train_df)

		#Create word2vec

		#create list of words with no repetitions
		all_words =[]
		for index, rows in clean_train_df.iterrows():
			all_words.append(rows['clean_tweet'].split(' '))
		flatlist_all = [item for sublist in all_words for item in sublist]
		single_list_of_words = list(set(flatlist_all))		

		#Word2Vec
		st.subheader("**_Word2Vec_**")
		st.info("Type in word from tweets that can be observed above")

		# Creating a text box for user input
		tweet_text_vec = st.text_area("Enter Text","Eg: realdonaldtrump")

		#Predict similar words
		if st.button('Predict Similar Words'):
			if tweet_text_vec in single_list_of_words:
				@st.cache(persist=True)
				def word2vec(text):
					model_w2v = Word2Vec(            
									tokenised_tweet,
									size=200, # desired no. of features/independent variables 
									window=5, # context window size
									min_count=2,
									sg = 1, # 1 for skip-gram model
									hs = 0,
									negative = 10, # for negative sampling
									workers= 2, # no.of cores
									seed = 34) 
					model_w2v.train(tokenised_tweet,total_examples= len(clean_train_df['clean_tweet']), epochs=20)	
					vec = model_w2v.wv.most_similar(positive=text)
					return vec
				predict_vec = word2vec(tweet_text_vec)
				for tuple in predict_vec:
					st.success("{}".format(tuple))
			else:
				st.success('Word Not found, please try again')

		#WordCloud Creation
		#Sentiment of 2
		# Create and generate a word cloud image:
		@st.cache(persist=True,allow_output_mutation=True)
		def WordCloud1(df):
			news_words =' '.join([text for text in df['clean_tweet'][df['sentiment'] == 2]])
			wordcloud = WordCloud(background_color ='white',width=2000, height=1500, random_state=21, max_font_size=300).generate(news_words)
			return wordcloud
		wordcloud1 = WordCloud1(clean_train_df)

		#Sentiment of 1
		# Create and generate a word cloud image:
		def WordCloud2(df):
			pro_words =' '.join([text for text in df['clean_tweet'][df['sentiment'] == 1]])
			wordcloud = WordCloud(background_color ='white',width=2000, height=1500, random_state=21, max_font_size=300).generate(pro_words)
			return wordcloud
		wordcloud2 = WordCloud2(clean_train_df)

		#Sentiment of 0
		# Create and generate a word cloud image:
		def WordCloud3(df):
			neutral_words =' '.join([text for text in df['clean_tweet'][df['sentiment'] == 0]])
			wordcloud = WordCloud(background_color ='white',width=2000, height=1500, random_state=21, max_font_size=300).generate(neutral_words)
			return wordcloud
		wordcloud3 = WordCloud3(clean_train_df)

		#Sentiment of -1
		# Create and generate a word cloud image:
		def WordCloud4(df):
			neg_words =' '.join([text for text in df['clean_tweet'][df['sentiment'] == 2]])
			wordcloud = WordCloud(background_color ='white',width=2000, height=1500, random_state=21, max_font_size=300).generate(neg_words)
			return wordcloud
		wordcloud4 = WordCloud4(clean_train_df)

		#Markdown for WordCloud
		st.subheader('**_WordCloud Plots_**')
		st.markdown('''
					<p>Plotting a <a href="https://www.geeksforgeeks.org/generating-word-cloud-python/" target="_blank">WordCloud</a> will help the common words used in a tweet. The most important analysis is understanding 
					sentiment and the wordcloud will show the common words used by looking at the train dataset</p>
					''', unsafe_allow_html=True)

		#Info
		st.info('WordClouds')
			
		if st.button("sentiment 2"):
			plt.imshow(wordcloud1)
			plt.axis("off")
			st.markdown("<h1 style='text-align: center; color: black;'> Word Cloud for News(2) Sentiment</h1>", unsafe_allow_html=True)
			plt.show()
			st.pyplot()
		if st.button("sentiment 1"):
			plt.imshow(wordcloud2)
			plt.axis("off")
			st.markdown("<h1 style='text-align: center; color: black;'> Word Cloud for Postive(1) Sentiment</h1>", unsafe_allow_html=True)
			plt.show()
			st.pyplot()
		if st.button('sentiment 0'):
			plt.imshow(wordcloud3)
			plt.axis("off")
			st.markdown("<h1 style='text-align: center; color: black;'> Word Cloud for Neutral(0) Sentiment</h1>", unsafe_allow_html=True)
			plt.show()
			st.pyplot()
		if st.button('sentiment -1'):
			plt.imshow(wordcloud4)
			plt.axis("off")
			st.markdown("<h1 style='text-align: center; color: black;'> Word Cloud for Negative(-1) Sentiment</h1>", unsafe_allow_html=True)
			plt.show()
			st.pyplot()	
		
		#Hashtags
		st.subheader('**_Hashtag Plots_**')
		st.markdown('''
					<p>The hashtags were plotted per sentiment as people use '#' in tweets 
					before a relevant keyword or phrase in their tweets.
					''', unsafe_allow_html=True)

		# function to collect hashtags
		@st.cache(persist=True,allow_output_mutation=True)
		def hashtag_extract(x):
			hashtags = []
			# Loop over the words in the tweet
			for i in x:
				ht = re.findall(r"#(\w+)", i)
				hashtags.append(ht)

			return hashtags	

		# extracting hashtags from  tweets
		HT_neutral = hashtag_extract(clean_train_df['clean_tweet'][clean_train_df['sentiment'] == 0])
		HT_pro = hashtag_extract(clean_train_df['clean_tweet'][clean_train_df['sentiment'] == 1])
		HT_news = hashtag_extract(clean_train_df['clean_tweet'][clean_train_df['sentiment'] == 2])
		HT_anti = hashtag_extract(clean_train_df['clean_tweet'][clean_train_df['sentiment'] == -1])	

		# unnesting list
		HT_neutral = sum(HT_neutral,[])
		HT_pro = sum(HT_pro,[])
		HT_news = sum(HT_news,[])
		HT_anti = sum(HT_anti,[])	


		#Plotting Hashtags
		#Info
		st.info('Hashtags')
			
		if st.button("Sentiment 2"):
			@st.cache(persist=True,allow_output_mutation=True)
			def hashtag1(lst):
				a = nltk.FreqDist(lst)
				d = pd.DataFrame({'Hashtag': list(a.keys()),'Count': list(a.values())})
				# selecting top 5 most frequent hashtags     
				d = d.sort_values(by = 'Count',ascending = False)
				return d[0:5]
			hash1 = hashtag1(HT_news)
			st.markdown("<h1 style='text-align: center; color: black;'> Hashtag for News(2) Sentiment</h1>", unsafe_allow_html=True)
			sns.barplot(data=hash1, x= "Hashtag", y = "Count")
			st.pyplot()
		if st.button("Sentiment 1"):
			@st.cache(persist=True,allow_output_mutation=True)
			def hashtag2(lst):
				a = nltk.FreqDist(lst)
				d = pd.DataFrame({'Hashtag': list(a.keys()),'Count': list(a.values())})
				# selecting top 5 most frequent hashtags     
				d = d.sort_values(by = 'Count',ascending = False)
				return d[0:5]
			hash2 = hashtag2(HT_pro)
			st.markdown("<h1 style='text-align: center; color: black;'> Hashtag for Postive(1) Sentiment</h1>", unsafe_allow_html=True)
			sns.barplot(data=hash2, x= "Hashtag", y = "Count")
			st.pyplot()
		if st.button('Sentiment 0'):
			@st.cache(persist=True,allow_output_mutation=True)
			def hashtag3(lst):
				a = nltk.FreqDist(lst)
				d = pd.DataFrame({'Hashtag': list(a.keys()),'Count': list(a.values())})
				# selecting top 5 most frequent hashtags     
				d = d.sort_values(by = 'Count',ascending = False)
				return d[0:5]
			hash3 = hashtag3(HT_neutral)
			st.markdown("<h1 style='text-align: center; color: black;'> Hashtag for Neutral(0) Sentiment</h1>", unsafe_allow_html=True)
			sns.barplot(data=hash3, x= "Hashtag", y = "Count")
			st.pyplot()
		if st.button('Sentiment -1'):
			@st.cache(persist=True,allow_output_mutation=True)
			def hashtag4(lst):
				a = nltk.FreqDist(lst)
				d = pd.DataFrame({'Hashtag': list(a.keys()),'Count': list(a.values())})
				# selecting top 5 most frequent hashtags     
				d = d.sort_values(by = 'Count',ascending = False)
				return d[0:5]
			hash4 = hashtag4(HT_anti)
			st.markdown("<h1 style='text-align: center; color: black;'> Hashtag for Negative(-1) Sentiment</h1>", unsafe_allow_html=True)
			sns.barplot(data=hash4, x= "Hashtag", y = "Count")
			st.pyplot()	


# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
