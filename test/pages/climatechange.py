import streamlit as st
import requests
import seaborn as sns
import pandas as pd
import numpy as np
import joblib,os

from PIL import Image
image = Image.open('climatechange.jpg')
st.image(image)
