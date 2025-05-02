# Used ChatGPT to help guide the writing and correction of the following code


import streamlit as st
import pandas as pd
from llm_selector import select_best_llm, call_llm_api


st.set_page_config(page_title="mAItcher", layout="wide")
st.subheader("We'll select the best AI for your prompt")

llm_classes = ['ChatGPT', 'Claude', 'Gemini', 'Mistral', 'Grok']
if 'llm_usage' not in st.session_state:
    st.session_state.llm_usage = {llm: 0 for llm in llm_classes}


if st.button("Submit"):
    with st.spinner("Selecting best LLM..."):
        selected_llm, predicted_label = select_best_llm(prompt)
        st.success(f"✅ Selected LLM: {selected_llm} (score: {predicted_label:.2f})")
 
        response = call_llm_api(prompt, selected_llm)
        st.markdown("### 🤖 Response")
        st.write(response)


if user_prompt:
    selected_llm, score = select_best_llm(user_prompt)
    st.write(f"**{selected_llm}** (score {score:.2f}) →")
    response = call_llm_api(user_prompt, selected_llm)
    st.write(response)


st.subheader('LLM Usage Statistics')
usage_df = pd.DataFrame.from_dict(
    st.session_state.llm_usage, 
    orient = 'index', 
    columns = ['Count']
)
st.bar_chart(usage_df)
prompt = st.text_area("Enter your prompt:", height=150)
 







