# ChatGPT wurde zur Korrektur, dem de-bugging und zur Anleitung für den folgenden Codes verwendet.


import streamlit as st # Um über Streamlit abrufbar zu sein.
import pandas as pd # Für Funktionen.
from llm_selector import select_best_llm, call_llm_api # Werden ebenfalls benötigt um die Chat-Funktion darstellen zu können.


# Konfiguration der Seite:
st.set_page_config(page_title="mAItcher", layout="wide")
st.title("mAItcher")
st.subheader("Enter your prompt and I'll pick the best LLM for you!")

# Quellen:
    # Streamlit. (o.D.-a). st.set_page_config. Streamlit documentation. Agerufen am 3. Mai, 2025, von https://docs.streamlit.io/develop/api-reference/configuration/st.set_page_config 
    # Streamlit. (o.D.-b). st.title. Streamlit documentation. Abgerufen am 3. Mai, 2025, von https://docs.streamlit.io/develop/api-reference/text/st.title
    # Streamlit. (o.D.-c). st.subheader. Streamlit documentation. Abgerufen am 3. Mai, 2025, von https://docs.streamlit.io/1.21.0/develop/api-reference/text/st.subheader 


llm_classes = ['ChatGPT', 'Claude', 'Gemini', 'Mistral', 'Grok'] # Wie im llm_selector, um weiter mit den Classes arbeiten zu können.
if 'llm_usage' not in st.session_state:
    st.session_state.llm_usage = {llm: 0 for llm in llm_classes} # Initialisiert den usage-counter.

# Quellen:
    # Streamlit, Inc. (o.D.-a). st.session_state. Streamlit documentation. Abgerufen am 3. Mai, 2025, von https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state
    # Streamlit, Inc. (o.D.-b). Session state: Architecture. Streamlit documentation. Abgerufen am 3. Mai, 2025, von https://docs.streamlit.io/develop/concepts/architecture/session-state
    # Python Software Foundation. (o.D.-a). Dictionaries. In The Python Tutorial. Abgerufen am 3. Mai, 2025, von https://docs.python.org/3/tutorial/datastructures.html#dictionaries
    # Python Software Foundation. (o.D.-b). Lists. In The Python Tutorial. Abgerufen am 3. Mai, 2025, von https://docs.python.org/3/tutorial/datastructures.html#more-on-lists


# If-Loop, damit die app sich merkt, welche LLM bisher verwendet wurde:
if 'last_selected' not in st.session_state: 
    st.session_state.last_selected = None #none, da es beim ersten Benutzen noch keine frühere Verwendung gab...
    st.session_state.last_response = None

# Quellen:
    # Streamlit, Inc. (o.D.-a). st.session_state. Streamlit documentation. Abgerufen am 3. Mai, 2025, von https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state
    # Streamlit, Inc. (o.D.-b). Session state: Architecture. Streamlit documentation. Abgerufen am 3. Mai, 2025, von https://docs.streamlit.io/develop/concepts/architecture/session-state


# Funktion, welche mit dem "Submit-Button" verbunden ist:
def on_submit():
    prompt = st.session_state.prompt  # wird von der from.-Funktion erfasst
    selected_llm, score = select_best_llm(prompt) # nimmt das beste LLM
    st.session_state.llm_usage[selected_llm] += 1 # fügt dem Usage-Zähler 1 Benutzung hinzu
    st.session_state.last_selected = f"{selected_llm} (score {score:.2f})"
    st.session_state.last_response = call_llm_api(prompt, selected_llm) # ruft die API des ausgewählten LLM auf und speichert deren Antwort

# Quellen:
    # Streamlit, Inc. (o.D.-a). st.session_state. Streamlit documentation. Abgerufen am 3. Mai, 2025, von https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state
    # Streamlit, Inc. (o.D.-b). Session state: Architecture. Streamlit documentation. Abgerufen am 3. Mai, 2025, von https://docs.streamlit.io/develop/concepts/architecture/session-state
    # Python Software Foundation. (o.D.-a). Defining functions. In The Python Tutorial. Abgerufen am 3. Mai, 2025, von https://docs.python.org/3/tutorial/controlflow.html#defining-functions
    # Python Software Foundation. (o.D.-b). Formatted string literals. In Lexical analysis. Abgerufen am 3. Mai, 2025, von https://docs.python.org/3/reference/lexical_analysis.html#f-strings


with st.form("prompt_form", clear_on_submit=False): #Erschafft die Box, in welche die User ihre Prompts schreiben können
    st.text_area("Your prompt:", key="prompt", height=150)
    st.form_submit_button("Submit", on_click=on_submit)

# Quellen:
    # Streamlit, Inc. (o.D.-a). st.form. Streamlit documentation. Abgerufen am 3. Mai, 2025, von https://docs.streamlit.io/develop/api-reference/widgets/st.form
    # Streamlit, Inc. (o.D.-b). st.text_area. Streamlit documentation. Abgerufen am 3. Mai, 2025, von https://docs.streamlit.io/develop/api-reference/widgets/st.text_area
    # Streamlit, Inc. (o.D.-c). st.form_submit_button. Streamlit documentation. Abgerufen am 3. Mai, 2025, von https://docs.streamlit.io/develop/api-reference/widgets/st.form_submit_button


if st.session_state.last_selected: # Zeigt, welches LLM ausgewählt wurde...
    st.success(f"Selected LLM: **{st.session_state.last_selected}**")
    st.markdown("###Response") 
    st.write(st.session_state.last_response) # ...und zeigt die Antwort an

# Quellen:
    # Streamlit, Inc. (o.D.-a). st.success. Streamlit documentation. Abgerufen am 3. Mai, 2025, von https://docs.streamlit.io/develop/api-reference/status/st.success
    # Streamlit, Inc. (o.D.-b). st.markdown. Streamlit documentation. Abgerufen am 3. Mai, 2025, von https://docs.streamlit.io/develop/api-reference/text/st.markdown
    # Streamlit, Inc. (o.D.-c). st.write. Streamlit documentation. Abgerufen am 3. Mai, 2025, von https://docs.streamlit.io/develop/api-reference/write-endpoints/st.write



# Funktion, um die Visualiserung schliesslich anzuzeigen:
st.subheader("LLM Usage Statistics")
usage_df = pd.DataFrame.from_dict(
    st.session_state.llm_usage,
    orient="index",
    columns=["Count"]
)
st.bar_chart(usage_df)

# Quellen:
    # Streamlit, Inc. (o.D.-a). st.subheader. Streamlit documentation. Abgerufen am 3. Mai, 2025, von https://docs.streamlit.io/develop/api-reference/text/st.subheader
    # Streamlit, Inc. (o.D.-b). st.bar_chart. Streamlit documentation. Abgerufen am 3. Mai, 2025, von https://docs.streamlit.io/develop/api-reference/charts/st.bar_chart
    # Pandas Development Team. (o.D.). pandas.DataFrame.from_dict. Pandas documentation. Abgerufen am 3. Mai, 2025, von https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.from_dict.html

