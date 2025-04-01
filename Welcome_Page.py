import streamlit as st

st.title("Nice to see you! Ready to be mAItched?")
st.write("Welcome to mAItcher, your connective AI-platform.")
st.header("How it works:")
st.write("Instead of typing your prompt into every different AI-chatbot you know and then deciding which answer is best, you can just give your prompt to me. I will mAItch you with the best fit model for your request.")

st.page_link("pages/mAitcher.py", label = "Get started", icon = "💫")

with st.sidebar:
  mAitcher = st.page_link("pages/mAitcher.py")
  chatGPT = st.page_link("pages/ChatGPT.py")
  Claude = st.pages_link("pages/Claude.py")
  Gemini = st.pages_link("pages/Gemini.py")
  Grok = st.pages_link("pages/Grok.py")
  Le_Chat = st.pages_link("pages/Le_Chat.py")

  st.navigation({
    "Choose your AI" : [mAitcher, chatGPT, Claude, Gemini, Grok, Le_Chat],
    "User Profile" : [My_Data, Use_Frequency]
  })
  
