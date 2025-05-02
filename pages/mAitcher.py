# Used ChatGPT to help guide the writing and correction of the following code

# For this page, we will need Streamlit, Pandas, as well as pull 2 functions from
# the file llm_selector.py.


import streamlit as st
import pandas as pd
from llm_selector import select_best_llm, call_llm_api


# Now, we need to configure the overall page settings.

# The sources:
    # Streamlit. (n.d.-a). st.set_page_config. Streamlit 
    # documentation. Retrieved May 3, 2025, from 
    # https://docs.streamlit.io/develop/api-reference/configuration/st.set_page_config 

    # Streamlit. (n.d.-b). st.title. Streamlit documentation. 
    # Retrieved May 3, 2025, from 
    # https://docs.streamlit.io/develop/api-reference/text/st.title

    # Streamlit. (n.d.-c). st.subheader. Streamlit documentation. 
    # Retrieved May 3, 2025, from 
    # https://docs.streamlit.io/1.21.0/develop/api-reference/text/st.subheader 


st.set_page_config(page_title="mAItcher", layout="wide")
st.title("mAItcher")
st.subheader("Enter your prompt and I'll pick the best LLM for you!")


# We now need to lay the foundations of the visualization chart.
# We also initialize the usage counter.

# The sources:
    # Streamlit, Inc. (n.d.-a). st.session_state. Streamlit documentation. 
    # Retrieved May 3, 2025, from 
    # https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state

    # Streamlit, Inc. (n.d.-b). Session state: Architecture. Streamlit documentation. 
    # Retrieved May 3, 2025, from 
    # https://docs.streamlit.io/develop/concepts/architecture/session-state

    # Python Software Foundation. (n.d.-a). Dictionaries. 
    # In The Python Tutorial. Retrieved May 3, 2025, from 
    # https://docs.python.org/3/tutorial/datastructures.html#dictionaries

    # Python Software Foundation. (n.d.-b). Lists. 
    # In The Python Tutorial. Retrieved May 3, 2025, from 
    # https://docs.python.org/3/tutorial/datastructures.html#more-on-lists


llm_classes = ['ChatGPT', 'Claude', 'Gemini', 'Mistral', 'Grok']
if 'llm_usage' not in st.session_state:
    st.session_state.llm_usage = {llm: 0 for llm in llm_classes}


# We also need to ensure that our app remembers which LLM was previously used
# to keep the tracker up to date even if the page is reloaded.
# Obviously, the first time we load the page, 'last_selected' and 
# 'last_response' are set to none because the user still hasn't interacted with
# our app.

# The Sources:
    # Streamlit, Inc. (n.d.-a). st.session_state. Streamlit documentation. 
    # Retrieved May 3, 2025, from 
    # https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state

    # Streamlit, Inc. (n.d.-b). Session state: Architecture. 
    # Streamlit documentation. Retrieved May 3, 2025, from 
    # https://docs.streamlit.io/develop/concepts/architecture/session-state


if 'last_selected' not in st.session_state:
    st.session_state.last_selected = None
    st.session_state.last_response = None


# 'on_submit' runs whenever the user hits the submit button. It takes the
# user's prompt, picks the best LLM and its score, and it increases the usage
# counter by 1 in the picked LLM column of the chart. Then, it calls
# the chosen LLM's API and finally saves its answer.

# The sources:
    # Streamlit, Inc. (n.d.-a). st.session_state. Streamlit documentation. 
    # Retrieved May 3, 2025, from 
    # https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state

    # Streamlit, Inc. (n.d.-b). Session state: Architecture. Streamlit documentation. 
    # Retrieved May 3, 2025, from 
    # https://docs.streamlit.io/develop/concepts/architecture/session-state

    # Python Software Foundation. (n.d.-a). Defining functions. 
    # In The Python Tutorial. Retrieved May 3, 2025, from 
    # https://docs.python.org/3/tutorial/controlflow.html#defining-functions

    # Python Software Foundation. (n.d.-b). Formatted string literals. 
    # In Lexical analysis. Retrieved May 3, 2025, from 
    # https://docs.python.org/3/reference/lexical_analysis.html#f-strings


def on_submit():
    prompt = st.session_state.prompt  # grabbed by the form below
    selected_llm, score = select_best_llm(prompt)
    st.session_state.llm_usage[selected_llm] += 1
    st.session_state.last_selected = f"{selected_llm} (score {score:.2f})"
    st.session_state.last_response = call_llm_api(prompt, selected_llm)


# The next block of code is the form inside the Streamlit app, where the user
# will type its prompt and hit submit.

# The sources:
    # Streamlit, Inc. (n.d.-a). st.form. Streamlit documentation. 
    # Retrieved May 3, 2025, from 
    # https://docs.streamlit.io/develop/api-reference/widgets/st.form

    # Streamlit, Inc. (n.d.-b). st.text_area. Streamlit documentation. 
    # Retrieved May 3, 2025, from 
    # https://docs.streamlit.io/develop/api-reference/widgets/st.text_area

    # Streamlit, Inc. (n.d.-c). st.form_submit_button. 
    # Streamlit documentation. Retrieved May 3, 2025, from 
    # https://docs.streamlit.io/develop/api-reference/widgets/st.form_submit_button


with st.form("prompt_form", clear_on_submit=False):
    st.text_area("Your prompt:", key="prompt", height=150)
    st.form_submit_button("Submit", on_click=on_submit)


# Once the user presses the submit button, this block shows which model
# got selected, creates a response header, and shows the AI's answer.

# The sources:
    # Streamlit, Inc. (n.d.-a). st.success. Streamlit documentation. 
    # Retrieved May 3, 2025, from 
    # https://docs.streamlit.io/develop/api-reference/status/st.success

    # Streamlit, Inc. (n.d.-b). st.markdown. Streamlit documentation. 
    # Retrieved May 3, 2025, from 
    # https://docs.streamlit.io/develop/api-reference/text/st.markdown

    # Streamlit, Inc. (n.d.-c). st.write. Streamlit documentation. 
    # Retrieved May 3, 2025, from 
    # https://docs.streamlit.io/develop/api-reference/write-endpoints/st.write


if st.session_state.last_selected:
    st.success(f"Selected LLM: **{st.session_state.last_selected}**")
    st.markdown("###Response")
    st.write(st.session_state.last_response)


# The last block of code takes the count of each LLM's usage and turns
# it into a chart spreadsheet.

# The sources:
    # Streamlit, Inc. (n.d.-a). st.subheader. Streamlit documentation. 
    # Retrieved May 3, 2025, from 
    # https://docs.streamlit.io/develop/api-reference/text/st.subheader

    # Streamlit, Inc. (n.d.-b). st.bar_chart. Streamlit documentation. 
    # Retrieved May 3, 2025, from 
    # https://docs.streamlit.io/develop/api-reference/charts/st.bar_chart

    # Pandas Development Team. (n.d.). pandas.DataFrame.from_dict. 
    # Pandas documentation. Retrieved May 3, 2025, from 
    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.from_dict.html


st.subheader("LLM Usage Statistics")
usage_df = pd.DataFrame.from_dict(
    st.session_state.llm_usage,
    orient="index",
    columns=["Count"]
)
st.bar_chart(usage_df)

