import streamlit as st

# Used ChatGPT to help guide the writing and correction of the following code

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

from llm_selector import select_best_llm, call_llm_api
 
st.title("🧠 Best LLM Selector")
 
prompt = st.text_area("Enter your prompt:", height=150)
 
if st.button("Submit"):
    with st.spinner("Selecting best LLM..."):
        selected_llm, predicted_label = select_best_llm(prompt)
        st.success(f"✅ Selected LLM: {selected_llm} (score: {predicted_label:.2f})")
 
        response = call_llm_api(prompt, selected_llm)
        st.markdown("### 🤖 Response")
        st.write(response)
