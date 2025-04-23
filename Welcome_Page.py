# Startseite für die App, welche die Funktion für User:innen erklärt
import streamlit as st

st.title("Nice to see you! Ready to be mAitched?")

st.markdown('''
    :rainbow[Welcome to mAItcher, your connective AI-platform.]''') #färbt den Text regenbogenfarbig für Ästethische Zwecke, Idee zum Code wurde der streamlit documentation entnommen

st.markdown('''
    :violet[**How it works:**] ''') #färbt den Text lila und macht ihn bold

st.markdown('''
    Instead of typing your prompt into every different AI-chatbot you know and then deciding which answer is best, you can just give your prompt to me. I will mAItch you with the 
    :orange-background[:violet[best fit model]] for your request.''') #hebt den wichtigsten Teil der Aussage mit orangenem Hintergrund und lila Schrift hervor

st.page_link("pages/mAitcher.py", label = "Get started", icon = "💫") #Button zur Interaktonsseite mit mAItcher, aus Streamlit Dokumentation
