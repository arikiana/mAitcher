# Used ChatGPT to help guide the writing and correction of the following code


import streamlit as st


st.set_page_config(
    page_title="mAItcher Welcome",
    layout="wide"
)


st.title("Nice to see you! Ready to be mAitched?")


tab1, tab2 = st.tabs(["Welcome to mAItcher", "Explore AI Models"])


with tab1:
    st.markdown(":rainbow[**Welcome to mAItcher, your connective AI-platform.**]")
    st.markdown(
        ":violet[**How it works:**]  "
        "Instead of typing your prompt into every different AI-chatbot and then deciding which answer is best, "
        "just give your prompt to mAItcher.  "
        "I’ll mAItch you with the :orange-background[:violet[best fit model]] for your request."
    )
    st.page_link(
        "pages/mAitcher.py",
        label="💫 Click here to get started",
        icon="💫"
    )


with tab2:
    st.subheader("Links to AI Platforms")
    st.markdown("""
    - 🌐 [ChatGPT](https://chatgpt.com/)  
    - 🚀 [Gemini](https://gemini.google.com/)  
    - 🐯 [Mistral](https://mistral.ai/)  
    - 🪐 [Grok](https://grok.com/)  
    - 👾 [Claude](https://claude.ai/)  
    """)

