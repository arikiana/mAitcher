# Used ChatGPT to help guide the writing and correction of the following code


import streamlit as st
import pandas as pd
from llm_selector import select_best_llm, call_llm_api


st.set_page_config(page_title="mAItcher", layout="wide")
st.title("mAItcher")
st.subheader("Enter your prompt and I'll pick the best LLM for you!")


llm_classes = ['ChatGPT', 'Claude', 'Gemini', 'Mistral', 'Grok']
if 'llm_usage' not in st.session_state:
    st.session_state.llm_usage = {llm: 0 for llm in llm_classes}


if 'last_selected' not in st.session_state:
    st.session_state.last_selected = None
    st.session_state.last_response = None


def on_submit():
    prompt = st.session_state.prompt  # grabbed by the form below
    selected_llm, score = select_best_llm(prompt)

    # increment counter
    st.session_state.llm_usage[selected_llm] += 1

    # store results for display
    st.session_state.last_selected = f"{selected_llm} (score {score:.2f})"
    st.session_state.last_response = call_llm_api(prompt, selected_llm)


with st.form("prompt_form", clear_on_submit=False):
    st.text_area("Your prompt:", key="prompt", height=150)
    st.form_submit_button("Submit", on_click=on_submit)


if st.session_state.last_selected:
    st.success(f"Selected LLM: **{st.session_state.last_selected}**")
    st.markdown("###Response")
    st.write(st.session_state.last_response)


st.subheader("LLM Usage Statistics")
usage_df = pd.DataFrame.from_dict(
    st.session_state.llm_usage,
    orient="index",
    columns=["Count"]
)
st.bar_chart(usage_df)







