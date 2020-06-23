import streamlit as st
import requests
import seaborn as sns
import pandas as pd
import numpy as np
import joblib,os

# Importing pages from the pages folder
import pages.climatechange
#import pages.trends

## Setting up the pages
PAGES = {
    "Overview": pages.climatechange
}

# Radio selector to switch over the different pages
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
