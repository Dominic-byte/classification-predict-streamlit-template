import streamlit as st
import requests
import seaborn as sns
import pandas as pd
import numpy as np
import joblib,os

### Title of climate change page
st.title("Climate Change")

### Added an image as a header
st.image("https://www.enr.com/ext/resources/Issues/National_Issues/2018/10-October/29-Oct/ViewpointBillWGettyImages-174525514_ENRwebready.jpg?1540410516", use_column_width= True)

### Describing climate climate
st.header("**_The terms “climate change” and “global warming” are used interchangeably but they mean separate things_**")
st.image("https://snowbrains.com/wp-content/uploads/2019/08/climate_change_buzzwords.jpg?w=640", use_column_width= True)

st.subheader(":bulb: What is climate change? :bulb:")
st.markdown("A change in global or regional climate patterns, in particular a change apparent from the mid to late 20th century onwards and attributed largely to the increased levels of atmospheric carbon dioxide produced by the use of fossil fuels")
