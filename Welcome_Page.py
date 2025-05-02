
# Startseite für die App, welche die Funktion für User:innen erklärt
import streamlit as st
import numpy as np
import pandas as pd

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


llm_classes = ['ChatGPT', 'Claude', 'Gemini', 'Mistral', 'Grok']
user_prompt = st.text_input("Enter your prompt:")
if 'llm_usage' not in st.session_state:
    st.session_state.llm_usage = {llm: 0 for llm in llm_classes}

if user_prompt:
    selected_llm, score = select_best_llm(user_prompt)
    st.write(f"**{selected_llm}** (score {score:.2f}) →")
    response = call_llm_api(user_prompt, selected_llm)
    st.write(response)
    
from llm_selector import select_best_llm, call_llm_api
st.subheader('LLM Usage Statistics')
usage_df = pd.DataFrame.from_dict(
    st.session_state.llm_usage, 
    orient = 'index', 
    columns = ['Count']
)
st.bar_chart(usage_df)


#Links um direkt auf die AIs zugreifen zu können, Funktionen via Streamlit Documentation 
tab2.subheader("Links")
tab2._link_button('ChatGPT', 'https://chatgpt.com/', "primary", icon="🌐")
tab2._link_button('Gemini', 'https://gemini.google.com/', "primary", icon="🚀")
tab2._link_button('Mistral', 'https://mistral.ai/', "primary", icon="🐯")
tab2._link_button('Grok','https://grok.com/', "primary", icon="🪐")
tab2._link_button('Claude', 'https://claude.ai/', "primary", icon="👾")
