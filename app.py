import streamlit as st
import joblib
import re
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Download NLTK files (for Streamlit cloud)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# Load saved model and vectorizer

model = joblib.load(
    "resume_naive_bayes_model.pkl"
)

tfidf = joblib.load(
    "resume_tfidf_vectorizer.pkl"
)


# Streamlit title

st.title("📄 Resume NLP Classification")
st.write(
    "Using NLP Techniques + Naive Bayes Algorithm"
)


# Text preprocessing function

def preprocess_text(text):

    # lowercase
    text = text.lower()


    # remove special characters
    text = re.sub(
        '[^a-zA-Z]',
        ' ',
        text
    )


    # tokenization
    tokens = word_tokenize(text)


    # stopword removal
    stop_words = set(
        stopwords.words('english')
    )


    filtered_words = []

    for word in tokens:
        if word not in stop_words:
            filtered_words.append(word)


    # lemmatization
    lemmatizer = WordNetLemmatizer()


    lemmatized_words = []

    for word in filtered_words:
        lemmatized_words.append(
            lemmatizer.lemmatize(word)
        )


    return " ".join(lemmatized_words)



# User input

resume_input = st.text_area(
    "Paste your Resume Text here",
    height=250
)


# Prediction button

if st.button("Predict Resume Category"):

    if resume_input.strip() == "":
        st.warning(
            "Please enter resume text"
        )

    else:

        # preprocess
        cleaned_resume = preprocess_text(
            resume_input
        )


        # convert text to vector
        resume_vector = tfidf.transform(
            [cleaned_resume]
        )


        # prediction
        prediction = model.predict(
            resume_vector
        )


        st.success(
            f"Predicted Category: {prediction[0]}"
        )