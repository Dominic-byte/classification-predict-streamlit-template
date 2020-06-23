import streamlit as st
import requests
import seaborn as sns
import pandas as pd
import numpy as np
import joblib,os

# Importing pages from the pages folder
import pages.coverpage
import pages.climatechange
import pages.mission
import pages.eda

## Setting up the pages
PAGES = {
    "Cover Page": pages.coverpage,
    "Overview": pages.climatechange,
    "Our Mission": pages.mission,
    "Exploratory Data Analysis": pages.eda
}

# Radio selector to switch over the different pages
selection = st.sidebar.radio("Go to", list(PAGES.keys()))

page = PAGES[selection]
