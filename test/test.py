### Libraries used in making of the web application
# Streamlit dependencies
import streamlit as st
import joblib, os

# Data dependencies
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple


### Loading the data
# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("https://raw.githubusercontent.com/Dominic-byte/classification-predict-streamlit-template/developing/resources/train.csv")

### The main function where we will build the actual app
def main():
    ## Creating sidebar with selection box
    # Creating load data
    st.sidebar.subheader(":heavy_check_mark: Data is loaded")
    st.sidebar.text_input("link to data", "https://raw.githubusercontent.com/Dominic-byte/classification-predict-streamlit-template/developing/resources/train.csv")

	# Creating multiple pages
    st.sidebar.title("Menu")
    options = ["Homepage", "Overview", "Our Mission", "Data Exploration", "Models", "About The Authors"]
    selection = st.sidebar.radio("Please select a page", options)

    # Creating about us box
    st.sidebar.title("About")
    st.sidebar.info("""
                    This web application is maintained by Three Musketeers. You can learn more about us on the **About The Authors** tab.
                    """)
    st.sidebar.image("https://i.pinimg.com/originals/5c/99/6a/5c996a625282d852811eac0ee9e81fbe.jpg",
                     use_column_width= True)

    ## Building our pages
    # Homepage page
    if selection == "Homepage":
        st.image("https://raw.githubusercontent.com/Dominic-byte/classification-predict-streamlit-template/developing/test/resources/coverpage.jpg",
                  use_column_width = True)


    # Overview page
    #if selection == "Overview":


    # Our Mission page
    #if selection == "Our Mission":


    # Data Exploration
    #if selection == "Data Exploration":


    # Models
    #if selection == "Models":


    # About the Authors
    #if selection == "About The Authors":







# Required to let Streamlit instantiate our web app.
if __name__ == '__main__':
	main()
