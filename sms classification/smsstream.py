
import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Load the model
model = load_model('sms_spam_model.h5')

# Load the vectorizer
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

st.title('SMS Spam Classifier')

st.write('Enter the SMS text below:')

# Input text
input_text = st.text_area("SMS Text", "")

# Predict if the input text is spam or ham
if st.button('Classify'):
    if input_text:
        input_vec = vectorizer.transform([input_text])
        prediction = model.predict(input_vec.toarray())[0][0]
        if prediction > 0.5:
            st.write('This SMS is *Spam*')
        else:
            st.write('This SMS is *not spam*')
    else:
        st.write('Please enter some text to classify.')
