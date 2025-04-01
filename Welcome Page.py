import streamlit as st

st.title("Nice to see you! Ready to be mAItched?")
st.write("Welcome to mAItcher, your connective AI-platform.")
st.header("How it works:")
st.write("Instead of typing your prompt into every different AI-chatbot you know and then deciding which answer is best, you can just give your prompt to me. I will mAItch you with the best fit model for your request.")

st.page_link("pages/mAitcher.py", label = "Get started", icon = "💫")

mAitcher = st.page_link("pages/mAitcher.py")
chatGPT = st.page_link("pages/AI Interfaces/chatGPT.py")
Claude =                      

st.navigation({
  "Choose your AI" : [mAitcher, chatGPT, Claude, Gemini, Grok, Le_Chat],
  "User Profile" : [My_Data, Use_Frequency]
})
