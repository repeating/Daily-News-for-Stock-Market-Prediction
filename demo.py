import streamlit as st
from predict import text_to_emotion


st.title('Emotion classification from text')

st.write('Possible emotions are sadness, anger, love, surprise, fear and joy')

text = st.text_input('Enter a text', "I'm sad to be a data scientist")
st.write('Your emotion is:', text_to_emotion(text))