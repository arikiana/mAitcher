
# Startseite für die App, welche die Funktion für User:innen erklärt
import streamlit as st
import numpy as np

st.title("Nice to see you! Ready to be mAitched?")

st.markdown('''
    :rainbow[Welcome to mAItcher, your connective AI-platform.]''') #färbt den Text regenbogenfarbig für Ästethische Zwecke, Idee zum Code wurde der streamlit documentation entnommen

st.markdown('''
    :violet[**How it works:**] ''') #färbt den Text lila und macht ihn bold

st.markdown('''
    Instead of typing your prompt into every different AI-chatbot you know and then deciding which answer is best, you can just give your prompt to me. I will mAItch you with the 
    :orange-background[:violet[best fit model]] for your request.''') #hebt den wichtigsten Teil der Aussage mit orangenem Hintergrund und lila Schrift hervor

st.page_link("pages/mAitcher.py", label = "Click here to get started", icon = "💫") #Button zur Interaktonsseite mit mAItcher, aus Streamlit Dokumentation

tab1, tab2 = st.tabs(["User Frequency", "Visit other AI-Models"]) #Tabs um die Visualisierung zu zeigen

tab1.subheader("Most popular AI-models")
tab1.text('We have tracked the most frequent matches between what our users search for and the different AIs. All relevant information can be found in the chart below:')

#Links um direkt auf die AIs zugreifen zu können, Funktionen via Streamlit Documentation 
tab2.subheader("Links")
tab2._link_button('ChatGPT', 'https://chatgpt.com/', "primary", icon="🌐")
tab2._link_button('Gemini', 'https://gemini.google.com/', "primary", icon="🚀")
tab2._link_button('Mistral', 'https://mistral.ai/', "primary", icon="🐯")
tab2._link_button('Grok','https://grok.com/', "primary", icon="🪐")
tab2._link_button('Claude', 'https://claude.ai/', "primary", icon="👾")
