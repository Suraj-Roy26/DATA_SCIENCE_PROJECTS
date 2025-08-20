import nltk
import string
from nltk.corpus import stopwords
import streamlit as st
import pickle
from nltk.stem.porter import PorterStemmer
import os
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # folder where app.py is located
tfidf_path = os.path.join(BASE_DIR, "vectorizer.pkl")
model_path = os.path.join(BASE_DIR, "model.pkl")

# Load pickled files
tfidf = pickle.load(open(tfidf_path, 'rb'))
model = pickle.load(open(model_path, 'rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_input("Enter the message")

if st.button('Predict'):

    # 1.preprocess

    transformed_sms = transform_text(input_sms)

    # 2.Vectorize

    vector_input = tfidf.transform([transformed_sms])

    # 3.Predict

    result = model.predict(vector_input)[0]

    # 4.Display

    if result == 1:
        st.header("Spam")
    else:
        st.header("Not spam")
