import streamlit as st

st.title("mAItcher")
prompt = st.chat_input("Write your prompt here")

if prompt:
    chat_box = f"""
    <div style='
        border 1px solid black; #HTML to create bubble like chat style, used ChatGPT to explain, navigate and correct code.
        padding: 10px;
        border-radius: 8px;
        background-color: #f0f0f0;
        margin: 10px;
        max-width: 80%
        width: fit-content
    '>

         You: {prompt}
    </div>
    """
    st.markdown (chat_box, unsafe_allow_html=True)

    
    st.markdown (chat_box, unsafe_allow_html=True)
