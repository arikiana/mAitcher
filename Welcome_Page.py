
# Startseite für die App, welche die Funktion für User:innen erklärt
import streamlit as st
import numpy as np
import pandas as pd

st.title("Nice to see you! Ready to be mAitched?")

#färbt den Text regenbogenfarbig für Ästethische Zwecke, Idee zum Code wurde der streamlit documentation entnommen
st.markdown(''':rainbow[Welcome to mAItcher, your connective AI-platform.]''')

#färbt den Text lila und macht ihn bold
st.markdown(''':violet[**How it works:**] ''') 

#hebt den wichtigsten Teil der Aussage mit orangenem Hintergrund und lila Schrift hervor
st.markdown(''' Instead of typing your prompt into every different AI-chatbot you know and then deciding which answer is best, you can just give your prompt to me. I will mAItch you with the :orange-background[:violet[best fit model]] for your request.''') 

#Button zur Interaktonsseite mit mAItcher, aus Streamlit Dokumentation
st.page_link("pages/mAitcher.py", label = "Click here to get started", icon = "💫") 

#Links um direkt auf die AIs zugreifen zu können, Funktionen via Streamlit Documentation 
tab2.subheader("Links")
tab2._link_button('ChatGPT', 'https://chatgpt.com/', "primary", icon="🌐")
tab2._link_button('Gemini', 'https://gemini.google.com/', "primary", icon="🚀")
tab2._link_button('Mistral', 'https://mistral.ai/', "primary", icon="🐯")
tab2._link_button('Grok','https://grok.com/', "primary", icon="🪐")
tab2._link_button('Claude', 'https://claude.ai/', "primary", icon="👾")
