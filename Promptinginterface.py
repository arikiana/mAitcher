import streamlit as st

st.title("mAItcher")
prompt = st.chat_input("Write your prompt here")
if prompt:
    st.write(f"You: {prompt}")
