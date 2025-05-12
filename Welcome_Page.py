# ChatGPT wurde zur Korrektur & dem de-bugging des folgenden Codes verwendet:


import streamlit as st

# Funktion zur Konfiguration auf streamlit: 
st.set_page_config(
    page_title="mAItcher Welcome",
    layout="wide"
)

# Quellen:
    # Streamlit. (o.D.). st.set_page_config. Streamlit documentation. Agerufen am 3. Mai, 2025, von https://docs.streamlit.io/develop/api-reference/configuration/st.set_page_config 


st.title("Nice to see you! Ready to be mAitched?") # Funktion zur Hervorhebung des Titels.

# Quellen:
    # Streamlit. (o.D.). st.title. Streamlit documentation. Abgerufen am 3. Mai, 2025, von https://docs.streamlit.io/develop/api-reference/text/st.title


tab1, tab2 = st.tabs(["Welcome to mAItcher", "Explore AI Models"]) # Zur Übersichtlichkeit wird die Homepage auf zwei Tabs aufgeteilt.

# Quellen:
    # Streamlit. (o.D.). st.tabs. Streamlit documentation. Abgerufen am 12. Mai, 2025, von https://docs.streamlit.io/develop/api-reference/layout/st.tabs


#Mit markdown wurde die Farbe der Texte sowie deren Format weiter verändert, um die Lesbarkeit des Textes zu steigern und wichtige Elemente hervorzuheben.
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
    ) # Button, der einen direkt mit dem Chatbot verbindet.

# Quellen:
    # Streamlit. (o.D. –a). st.tabs. Streamlit documentation. Abgerufen am 12. Mai, 2025, von https://docs.streamlit.io/develop/api-reference/layout/st.tabs
    # Streamlit. (o.D. –b). st.page_link. Streamlit documentation. Abgerufen am 12. Mai, 2025, von https://docs.streamlit.io/develop/api-reference/widgets/st.page_link
    # Streamlit. (o.D. –c). st.markdown. Streamlit documentation. Abgerufen am 12. Mai, 2025, von https://docs.streamlit.io/develop/api-reference/text/st.markdown


# Um bei einer Präferenz für einzelne Ai-Platformen direkt darauf zugreifen zu können, werden die verwendeten Modelle separat verlinkt:
with tab2:
    st.markdown(":rainbow[**Links to AI Platforms**]")
    st.markdown("""
    - 🌐 [ChatGPT](https://chatgpt.com/)  
    - 🚀 [Gemini](https://gemini.google.com/)  
    - 🐯 [Mistral](https://mistral.ai/)  
    - 🪐 [Grok](https://grok.com/)  
    - 👾 [Claude](https://claude.ai/)  
    """)

# Quellen:
    # Streamlit. (o.D. –a). st.tabs. Streamlit documentation. Abgerufen am 12. Mai, 2025, von https://docs.streamlit.io/develop/api-reference/layout/st.tabs
    # Streamlit. (o.D. –b). st.markdown. Streamlit documentation. Abgerufen am 12. Mai, 2025, von https://docs.streamlit.io/develop/api-reference/text/st.markdown
