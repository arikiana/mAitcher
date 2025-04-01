import streamlit as st

st.title("Nice to see you! Ready to be mAItched?")
st.write("Welcome to mAItcher, your connective AI-platform.")
st.header("How it works:")
st.write("Instead of typing your prompt into every different AI-chatbot you know and then deciding which answer is best, you can just give your prompt to me. I will mAItch you with the best fit model for your request.")

st.markdown("Nice to see you! Ready to be **mAItched?**")
st.markdown('''
    :rainbow[Welcome to mAItcher, your connective AI-platform.] :black[How it works:] 
    :black[Instead of typing your prompt into every different AI-chatbot you know and then deciding which answer is best, you can just give your prompt to me. I will mAItch you with the] 
    :blue-background[best fit model] :black[for your request.]''')

st.page_link("pages/mAitcher.py", label = "Get started", icon = "💫")

with st.sidebar:
  st.subheader("Choose your AI:")
  mAitcher = st.page_link("pages/mAitcher.py")
  chatGPT = st.page_link("pages/ChatGPT.py")
  Claude = st.page_link("pages/Claude.py")
  Gemini = st.page_link("pages/Gemini.py")
  Grok = st.page_link("pages/Grok.py")
  Le_Chat = st.page_link("pages/Le_Chat.py")

  
