import streamlit as st

st.title("Nice to see you! Ready to be mAitched?") 
st.markdown('''
    :rainbow[Welcome to mAItcher, your connective AI-platform.]''') #raibow markdown creates a colorful effect in the text, code sourced through streamlit discussion forum
st.markdown('''
    :blue[**How it works:**] ''') #blue markdown for blue text, code sourced through streamlit discussion forum

st.markdown('''
    Instead of typing your prompt into every different AI-chatbot you know and then deciding which answer is best, you can just give your prompt to me. I will mAItch you with the 
    :blue-background[best fit model] for your request.''') #hilights best fit in blue, code sourced through streamlit discussion forum

st.page_link("pages/mAitcher.py", label = "Get started", icon = "💫") #creates a Button to the interaction page with maitcher
