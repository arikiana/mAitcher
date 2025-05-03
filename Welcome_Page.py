# ChatGPT wurde zur Korrektur & dem debugging des folgenden Codes verwendet


import streamlit as st


st.set_page_config(
    page_title="mAItcher Welcome",
    layout="wide"
)


st.title("Nice to see you! Ready to be mAitched?") #Funktion aus der Streamlit Documentation zur Hervorhebung des Titels


tab1, tab2 = st.tabs(["Welcome to mAItcher", "Explore AI Models"]) #zur Übersichtlichkeit wird die Homepage auf zwei Tabs aufgeteilt, Funktion aus Streamlit Documentation

# mit markdown wurde die Farbe der Texte sowie deren Format weiter verändert, um die Lesbarkeit des Textes zu steigern und wichtige Elemente hervorzuheben (aus Streamlit Documentation)
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
        label="Click here to get started",
        icon="💫"
    ) #button, der einen direkt mit dem chatbot verbindet

# um bei einer Präferenz für einzelne Ai-Platformen direkt darauf zugreifen zu können, werden die verwendeten Modelle separat verlinkt
with tab2:
    st.markdown(":rainbow[**Links to AI Platforms**]")
    st.markdown("""
    - 🌐 [ChatGPT](https://chatgpt.com/)  
    - 🚀 [Gemini](https://gemini.google.com/)  
    - 🐯 [Mistral](https://mistral.ai/)  
    - 🪐 [Grok](https://grok.com/)  
    - 👾 [Claude](https://claude.ai/)  
    """)

