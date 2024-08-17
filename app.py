import streamlit as st
import pickle
import pandas as pd
import numpy as np
import nltk
import re
import string

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

nltk.download('stopwords')
nltk.download('punkt')

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))

def transform_text(text):
    ## Lowercasing the text
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    ## Remove Punctuations
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()

    ## Remove stopwords
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    ## Stemming
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)


st.title('Diagnosify')
st.header('Welcome to Diagnosify!! Your own disease Predictor')
input = st.text_area('Enter the news text', height=250)

if st.button('Predict'):
    
    # 1. Transforming the text
    transform_news = transform_text(input)
    
    # 2. Vectorize the given transform text
    vector_input = tfidf.transform([transform_news]).toarray()
    
    # 3. Make the prediction
    
    result = model.predict(vector_input)
    
    encoded_output = encoder.inverse_transform(result)
    
    st.header(encoded_output[0])    