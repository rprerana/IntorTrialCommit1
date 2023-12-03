import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import ktrain
from ktrain import text

st.subheader(':rainbow[From the above graph we can see that there is a difference in the number of reviews based on the polarity]') 

p = ktrain.load_predictor('drive/My Drive/m2bert.keras')

st.subheader(':rainbow[ModelLoaded]') 

test_data =['Barbie 2023 is a tour de force that has left me utterly captivated, enchanted, and spellbound. Every moment of this cinematic marvel was nothing short of pure excellence, deserving nothing less than a perfect 10 out of 10 rating!']
# sample test data prediction.
for a in p.predict(test_data):
  if a=='not_sentiment':
    st.write('NEGATIVE')
  else:
    st.write('POSITIVE')


