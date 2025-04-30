import streamlit as st

# Used ChatGPT to help guide the writing and correction of the following code

st.title("mAItcher")

from llm_selector import select_best_llm, call_llm_api
 
st.subheader("We'll select the best AI for your prompt")
 
prompt = st.text_area("Enter your prompt:", height=150)
 
if st.button("Submit"):
    with st.spinner("Selecting best LLM..."):
        selected_llm, predicted_label = select_best_llm(prompt)
        st.success(f"✅ Selected LLM: {selected_llm} (score: {predicted_label:.2f})")
 
        response = call_llm_api(prompt, selected_llm)
        st.markdown("### 🤖 Response")
        st.write(response)
