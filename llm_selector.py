# Used ChatGPT to help guide the writing and correction of the following code

# -------------------

# Our app project is to provide the end-user with one interface connected to
# the API of 5 LLMs (ChatGPT, Claude, Gemini, Le Chat, and Grok).

# The user writes a prompt and our linear regression machine learning model will
# analyze it and choose the best-fitted LLM to answer it.

# The app will then -through the chosen LLM's API- get and show the answer
# to the user.

# -------------------

# After extensive research on how to work with the dataset, we chose to create
# a .csv file in Excel which contains 4 categories of 10 prompts each.

# Each LLM received each prompt and we graded their answer on a scale from 0
# (worst) to 10 (best). We also created a column which indicates -in the
# integer form- which LLM gave the best answer and thus, is best fitted to
# provide future answer on similar prompts.

# The LLMs are listed as follows: ChatGPT is 0, Claude is 1, Gemini is 2,
# Le Chat is 3, and Grok is 4.

# As the app is developped in Streamlit, we need to import it in our code.
# We will also need NumPy later in our code.

# Once the .csv file was done, we made some extensive research to find the right
# Python library to work with our tabular dataset. We found out it is the Pandas
# library.

# The source:
    # pandas development team. (n.d.). Getting started. Pandas documentation.
    # Retrieved April 1, 2025, from
    # https://pandas.pydata.org/docs/getting_started/index.html#getting-started​


import streamlit as st
import pandas as pd
import numpy as np


# Once Pandas is imported, we need this library to interact with our .csv file.
# This can be done through Pandas' dataframe which we named -by convention- df.

# df then needs to interact with pd.read_csv to access our .csv file which is
# named "prompts.csv". It will read the .csv file into a DataFrame.

# The sources:
    # pandas development team. (n.d.). pandas.read_csv.
    # pandas documentation. Retrieved April 1, 2025, from
    # https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html

    # IPython development team. (n.d.). IPython.display.display. IPython
    # documentation. Retrieved April 1, 2025, from
    # https://ipython.readthedocs.io/en/8.26.0/api/generated/IPython.display.html

    # pandas development team. (n.d.). pandas.DataFrame. pandas documentation.
    # Retrieved April 1, 2025, from
    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html​


df = pd.read_csv("prompts.csv")


# We know need to train our model on our data set. After some research, it
# appeared that the sklearn library would be best to train our model.

# We also discovered that we needed to split our
# dataset into training and testing sets. Thus, we need to import the
# "train_test_split" module. This splits our data into a training set
# and a testing set.

# The sources:
    # Google Developers. (n.d.). Dividing the original dataset. In Machine
    # Learning Crash Course. Retrieved April 1, 2025, from
    # https://developers.google.com/machine-learning/crash-course/overfitting/dividing-datasets​

    # scikit-learn developers. (n.d.). Computing cross-validated metrics.
    # scikit-learn documentation. Retrieved April 1, 2025, from
    # https://scikit-learn.org/stable/modules/cross_validation.html#computing-cross-validated-metrics​

    # scikit-learn developers. (n.d.). sklearn.model_selection.train_test_split.
    # scikit-learn documentation. Retrieved April 1, 2025, from
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html​


from sklearn.model_selection import train_test_split


# After losing a lot of time, we finally came to the conclusion -and understood-
# that machine learning models can not work with text directly. Thus, we needed to
# import the TF-IDF Vectorizer which formats text into vectors. This way, our
# model can understand the data it has to train on as it transforms our text into
# numerical vectors.

# The sources:
    # scikit-learn developers. (n.d.). Text feature extraction.
    # scikit-learn documentation. Retrieved April 1, 2025, from
    # https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction

    # scikit-learn developers. (n.d.).
    # sklearn.feature_extraction.text.TfidfVectorizer. scikit-learn
    # documentation. Retrieved April 1, 2025, from
    # https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

    # Masudowolabi. (2024, June 10). How to use sklearn’s TfidfVectorizer for
    # text feature extraction in model testing. Medium. Retrieved April 1, 2025,
    # from https://medium.com/@masudowolabi/how-to-use-sklearns-tfidfvectorizer-for-text-feature-extraction-in-model-testing-e1221fd274f8


from sklearn.feature_extraction.text import TfidfVectorizer


# One might think that using the RandomForestClasifier would be best as it combines
# the prediction of multiple decision trees. However, after carefully analyzing the data, 
# we could see that each LLM responds in a
# rather predictable way to prompts' specific features (e.g. coding). Thus, one
# can see linear patterns between prompt features and LLM scores. This is why we
# we need to import the LinearRegression from sklearn.

# The sources:
    # scikit-learn developers. (n.d.). Linear models. scikit-learn
    # documentation. Retrieved April 1, 2025, from
    # https://scikit-learn.org/stable/modules/linear_model.html​

    # scikit-learn developers. (n.d.). sklearn.ensemble.RandomForestClassifier.
    # scikit-learn documentation. Retrieved April 1, 2025, from
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html​

    # scikit-learn developers. (n.d.). sklearn.linear_model.LinearRegression.
    # scikit-learn documentation. Retrieved April 1, 2025, from
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html


from sklearn.linear_model import LinearRegression


# After we'll have trained our model on our data, we will need a feedback loop
# to assess the quality of our data. To do so, we relied on our Operations
# Management knowledge. We decided to use the Mean Squared Error and the R2
# score. MSE measures the average squared difference between the predicted and
# actual value. The lower the better the performance of our model. R2 indicates how
# well the model explains the variability of the target variable. The closer the R2 score 
# is to 1.0, the more patterns the model captures.

#The source:
    # scikit-learn developers. (n.d.). sklearn.metrics.mean_squared_error.
    # scikit-learn documentation. Retrieved April 1, 2025, from
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html

    # scikit-learn developers. (n.d.). Mean squared error. In Metrics and
    # scoring: Quantifying the quality of predictions.
    # scikit-learn documentation. Retrieved April 1, 2025, from
    # https://scikit-learn.org/stable/modules/model_evaluation.html#mean-squared-error

    # scikit-learn developers. (n.d.). sklearn.metrics.r2_score.
    # scikit-learn documentation. Retrieved April 1, 2025, from
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html

    # scikit-learn developers. (n.d.). R² score. In Metrics and scoring:
    # Quantifying the quality of predictions. scikit-learn documentation.
    # Retrieved April 1, 2025, from
    # https://scikit-learn.org/stable/modules/model_evaluation.html#r2-score


from sklearn.metrics import mean_squared_error, r2_score


# Now that all the necessary and relevant tools have been imported, we will
# begin the training of our model with our .csv file.

# For the data to be used as "training food", we need to extract it from the
# .csv file. X extracts the input features; it is what the model will learn
# from. Y extracts the target values; basically, what the model is training to
# predict. The target value is the model that had the highest score on a
# particular prompt.

# The Sources:
    # scikit-learn developers. (n.d.). Feature selection. scikit-learn
    # documentation. Retrieved April 1, 2025, from
    # https://scikit-learn.org/stable/modules/feature_selection.html​

    # pandas development team. (n.d.). pandas.DataFrame. pandas documentation
    # (version 0.23). Retrieved April 1, 2025, from
    # https://pandas.pydata.org/pandas-docs/version/0.23/generated/pandas.DataFrame.html​


X = df['prompt_text']
y = df['best_model_label']


# As mentionned above, plain text cannot be used to train a model. Thus, we are going to
# vectorize our data to add a specific and granular importance on each words depending on
# the context.

# The source:
    # scikit-learn developers. (n.d.). Text feature extraction. In Feature
    # extraction. scikit-learn documentation. Retrieved April 1, 2025, from
    # https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction​


vectorizer = TfidfVectorizer()


# Now that the data has been vectorized,the following snippet allows us to make 
# the model learn the vocabulary and transform it into a matrix. 
# This way, it creates a matrix of the semantic text patterns numerically.

# The source:
    # scikit-learn developers. (n.d.).
    # sklearn.feature_extraction.text.CountVectorizer.
    # scikit-learn documentation. Retrieved April 1, 2025, from
    # https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html​


X_vect = vectorizer.fit_transform(X)


# At this stage, the data is almost fully ready to undergo the training.
# The last step is to split the data between training and testing sets.

# The code snippet is separated into the following parts:
  # X_train: the inputs the model will be trained on.
  # X_test: the inputs that the model is actually tested on.
  # y_train: the correct output for training.
  # y_test: the correct output for testing.

# The second part of the snippet is the train_test_split module and its
# parameters.
  # X_vect: the vectorized text data.
  # y: target values we want to predict.
  # test_size = 0.05: the ratio of the training and the testing data.
  # random_state = 21: controls how the data is shuffled before it's split.

# The X and the y are respectively coming from
# "X = df['prompt_text']" and "y = df['best_model_label']".

# For the test_size, we used 0.05 because it allows the model to train on 95% of
# the data and test on 5%. This split seems the most reasonable as we do not
# have a lot of data available. Thus, we want to maximize the training
# iterations.

# For the random_state, we randomly decided to take 21 as it is a random
# variable.

# The source:
    # scikit-learn developers. (n.d.).
    # sklearn.model_selection.train_test_split.
    # scikit-learn documentation. Retrieved April 1, 2025, from
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html​


X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size = 0.05, random_state = 21)


# The data is now ready to undergo the training and testing part. We are going to
# use the linear regression to draw a line through our data (linearise
# the data). This snippet acts as the initializer for the model.

# The source:
    # scikit-learn developers. (n.d.). sklearn.linear_model.LinearRegression.
    # scikit-learn documentation. Retrieved April 1, 2025, from
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html​


model = LinearRegression()


# The data has been linearised and is now ready to be trained by making 
# the input features (X_train) interact with the target values (y_train). 
# In other words, our model is learning the patterns through model.fit.

#The source:
    # scikit-learn developers. (n.d.). Cross-validation: evaluating estimator
    # performance. scikit-learn documentation. Retrieved April 1, 2025, from
    # https://scikit-learn.org/stable/modules/cross_validation.html​


model.fit(X_train, y_train)


# The model has now been trained and is ready to be tested on unseen data.
# For this part, y_pred is a list of predicted values which the model thinks
# are correct for each input in X_test

#The source:
    # Brownlee, J. (2020, January 10). How to make predictions with
    # scikit-learn. Machine Learning Mastery. Retrieved April 1, 2025, from
    # https://machinelearningmastery.com/make-predictions-scikit-learn/​


y_pred = model.predict(X_test)


# Now, that our data has been tested with the unseen data, one needs to get
# a feedback on the quality through the MSE and R2. Both snippets 
# are comparing the actual values (y_test) and the predicted values (y_pred).

# After discussing it with ChatGPT, an MSE of 0.876 and a r2 score of 0.610 are
# decent results for the amount of data we have. Thus, we can continue to the next step.

# The sources:
    # scikit-learn developers. (n.d.). Mean squared error. In Metrics and
    # scoring: quantifying the quality of predictions. scikit-learn
    # documentation. Retrieved April 1, 2025, from
    # https://scikit-learn.org/stable/modules/model_evaluation.html#mean-squared-error

    # scikit-learn developers. (n.d.). R² score. In Metrics and scoring:
    # quantifying the quality of predictions. scikit-learn documentation.
    # Retrieved April 1, 2025, from
    # https://scikit-learn.org/stable/modules/model_evaluation.html#r2-score


print(mean_squared_error(y_test, y_pred))
print(r2_score(y_test, y_pred))


# The training stage for our model is now complete.
# Now, one needs to pick the best LLM based on each user's prompt.
# Thus, we need to use the trained model to predict a score for each LLM.

# For this, we need to define the list of the LLMs that we are going to be using.
# The index of each model is positioned based on its label in the data set.


llm_classes = ['ChatGPT', 'Claude', 'Gemini', 'Mistral', 'Grok']


# Now, we need to define a function that will return a prompt and provide us
# back with the most adapted model to answer the user's prompt. 

# The following code snippet is designed to predict and select the best LLM for
# a given prompt. To do this, we have 5 other dependant functions.

# The first function is 'vect_prompt' which uses the TF-IDF to transform the user's
# prompt into numerical data based on the vocabulary used during training.

# Second, 'predicted_label' uses the trained model to predict the label based on
# the prompt. This will give us a float number which represents the regressed
# estimate of the best LLM for a particular prompt. The logic behind this is that 
# we will get a float number between 0 and 5 from the regression. If one remembers,
# our list of LLMs goes from 0 to 5. Thus, the float number we get corresponds to the 
# best fitted LLM to answer the prompt.

# Third, 'best_index' is rounding the float number to the nearest integer. This
# integer is the position of one of our LLM, meaning this LLM is the most fit
# to answer this particular prompt.

# Fourth, 'best_llm' uses the integer to retrieve the corresponding name of the
# best fit LLM from the 'llm_classes' list.

# Lastly, 'return' provides us with all the relevant information about our
# prediction

# Moreover, as we want the user to see how many times each LLM was selected, we
# need to track the number of times each LLM was selected. Then, this will
# be displayed in a bar chart. Now, we need to update the usage tracker 
# as the LLM has been selected.

# The sources:
    # Python Software Foundation. (n.d.). More on lists. In The Python tutorial.
    # Retrieved April 1, 2025, from
    # https://docs.python.org/3/tutorial/datastructures.html#more-on-lists​

    # NumPy developers. (n.d.). What is NumPy?. NumPy documentation.
    # Retrieved April 1, 2025, from
    # https://numpy.org/devdocs/user/whatisnumpy.html​

    # Python Software Foundation. (n.d.). Defining functions.
    # In The Python tutorial. Retrieved April 1, 2025, from
    # https://docs.python.org/3/tutorial/controlflow.html#defining-functions

    # scikit-learn developers. (n.d.).
    # sklearn.feature_extraction.text.TfidfVectorizer. scikit-learn
    # documentation. Retrieved April 1, 2025, from
    # https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

    # scikit-learn developers. (n.d.).
    # sklearn.linear_model.LinearRegression.predict. scikit-learn documentation.
    # Retrieved April 1, 2025, from
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression.predict

    # Python Software Foundation. (n.d.). round. In Built-in functions.
    # Retrieved April 1, 2025, from
    # https://docs.python.org/3/library/functions.html#round

    # Python Software Foundation. (n.d.). Lists. In An informal introduction to
    # Python. Retrieved April 1, 2025, from
    # https://docs.python.org/3/tutorial/introduction.html#lists

    # Python Software Foundation. (n.d.). Dictionaries. In Data structures.
    # Retrieved April 1, 2025, from
    # https://docs.python.org/3/tutorial/datastructures.html#dictionaries

    # Streamlit, Inc. (n.d.-a). st.session_state. Streamlit documentation. 
    # Retrieved May 2, 2025, from 
    # https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state


def select_best_llm(prompt):
    vect_prompt    = vectorizer.transform([prompt])
    predicted_label = model.predict(vect_prompt)[0]
    best_index      = round(predicted_label)
    selected_llm    = llm_classes[best_index]
    return selected_llm, predicted_label

    
    if 'llm_usage' not in st.session_state:
        st.session_state.llm_usage = {llm: 0 for llm in llm_classes}
    st.session_state.llm_usage[choice] += 1
    return choice, pred


# We are now tackling the last part of our project. We need to connect our code
# to the model's API to get the answer to our prompt from the selected LLM. 
# Our code detects which LLM should be called and it sends the prompt's request 
# to the correct LLMs API. Then, the LLM answers the user's prompt and sends 
# the answer back to the user's screen.

# To be able to "call" the correct LLM based on our trained model through its API,
# we need to create a logic pathway of which LLM's API our code needs 
# to contact. Our code contatcs the correct API based on the integer the 
# user's prompt created.

# The logic is the following: if the 'selected_llm' variable is equal to 'LLM',
# then, it calls the function 'call_llm(prompt)' to send the prompt to the LLM's
# API. It basically centralizes the logic for sending prompts to the correct LLM.


def call_llm_api(prompt, selected_llm):
  if selected_llm == 'ChatGPT':
    return call_chatgpt(prompt)
  elif selected_llm == 'Claude':
    return call_claude(prompt)
  elif selected_llm == 'Gemini':
    return call_gemini(prompt)
  elif selected_llm == 'Mistral':
    return call_mistral(prompt)
  elif selected_llm == 'Grok':
    return call_grok(prompt)
  else:
    return "Error: Unknown LLM selected"


# Now, that the logic pathway has been created, we will write the code that will
# call OpenAI's ChatGPT API if the float number is nearest to 0 (ChatGPT's place being 0).
# The code is based on OpenAI's own documentation as well as our own
# modifications to make the code work in our workflow.

# The sources:
    # OpenAI. (n.d.). OpenAI Python library. OpenAI API documentation.
    # Retrieved April 1, 2025, from
    # https://platform.openai.com/docs/libraries/python-library?language=python

    # OpenAI. (n.d.). Step 2: Set up your API key. OpenAI API documentation.
    # Retrieved April 1, 2025, from
    # https://platform.openai.com/docs/quickstart/step-2-set-up-your-api-key?api-mode=responses

    # Python Software Foundation. (n.d.). Defining functions. In The Python
    # tutorial. Retrieved April 1, 2025, from
    # https://docs.python.org/3/tutorial/controlflow.html#defining-functions


import openai
client = openai.OpenAI(api_key=st.secrets["openai"]["api_key"])
def call_chatgpt(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"ChatGPT API error: {str(e)}"


# This part of the code will call Claude's API if prompted by the user's prompt.
# The code is based on Anthropic's own documentation as well as our own
# modifications to make the code work in our workflow.

# The sources:
    # Anthropic. (n.d.). Anthropic documentation.
    # Retrieved April 1, 2025, from https://docs.anthropic.com/en/home

    # Anthropic. (n.d.). Messages API. Retrieved April 1, 2025,
    # from https://docs.anthropic.com/en/api/messages

    # Anthropic. (n.d.). Getting started. Retrieved April 1, 2025,
    # from https://docs.anthropic.com/en/api/getting-started

    # Anthropic. (n.d.). Initial setup. Retrieved April 1, 2025,
    # from https://docs.anthropic.com/en/docs/initial-setup


import anthropic
claude_client = anthropic.Anthropic(api_key=st.secrets["anthropic"]["claude_api_key"])
def call_claude(prompt: str) -> str:
    try:
        response = claude_client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=1000,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text
    except Exception as e:
        return f"Claude API error: {e}"


# This part of the code will call Gemini's API if prompted by the user's prompt.
# The code is based on Google's own documentation as well as our own
# modifications to make the code work in our workflow.

# The sources:
    # Google. (n.d.). Gemini API documentation. Retrieved April 1, 2025,
    # from https://ai.google.dev/gemini-api/docs

    # Google. (n.d.). Gemini API quickstart. Retrieved April 1, 2025,
    # from https://ai.google.dev/gemini-api/docs/quickstart

    # Google. (n.d.). Gemini models. Retrieved April 1, 2025,
    # from https://ai.google.dev/gemini-api/docs/models

    # Google. (n.d.). Gemini API available regions. Retrieved April 1, 2025,
    # from https://ai.google.dev/gemini-api/docs/available-regions


import google.generativeai as genai
from google.generativeai import GenerativeModel
genai.configure(api_key = "AIzaSyBbsMHZTSKv2BF4Rw2KOMWeHcb4RtWzZiA")
def call_gemini(prompt):
    try:
        model = GenerativeModel(model_name = "gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini API error: {str(e)}"


# This part of the code will call Mistral's API if prompted by the user's prompt.
# The code is based on Mistral's own documentation as well as our own
# modifications to make the code work in our workflow.

# The sources:
    # Mistral AI. (n.d.). Mistral AI documentation.
    # Retrieved April 1, 2025, from https://docs.mistral.ai/

    # Mistral AI. (n.d.). Mistral AI API.
    # Retrieved April 1, 2025, from https://docs.mistral.ai/api/


from mistralai import Mistral
mistral_client = Mistral(api_key = "vJQyWusvYsujsVZDZZiUjaSLDjbV8H4C")
def call_mistral(prompt: str) -> str:
    try:
        response = mistral_client.chat.complete(
            model = "mistral-small",
            messages = [{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    except Exception as e:
        response = mistral_client.chat.complete(
            model = "mistral-small",
            messages = [{"role": "user", "content": prompt}],
        )


# This part of the code will call XAI's API if prompted by the user's prompt.
# The code is based on OpenAI's own documentation and xAI's own documentation
# as well as our own modifications to make the code work in our workflow. xAI
# still relies on OpenAI's infrastructure to use its API in Python.

# The sources:
    # moesmufti. (n.d.). xai_grok_sdk [Computer software].
    # GitHub. Retrieved April 1, 2025, from
    # https://github.com/moesmufti/xai_grok_sdk


from openai import OpenAI
client = OpenAI(
    api_key = "xai-eQkIynQcrNUXG5CN1WskQnn9hcegTV1PqDSHb2k0Rb4NOq9dSVhx4kDITUx7WKXte7uENQcoGFzZaOO5",
    base_url = "https://api.x.ai/v1"
)
def call_grok(prompt: str) -> str:
    try:
        resp = client.chat.completions.create(
            model = "grok-2-1212",
            messages = [{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"Grok error: {e}"


# Now, we want to be able to retrieve the user's prompt, have the best fitted
# LLM to handle that prompt, call that LLM via its API, and display its response.

# First, we want to get the user's prompt with 'prompt'.
# Second, we call 'select_best_llm' to return the name of the best model.
# Third, we display the selected LLM and the numeric output (the float number).
# Fourth, 'llm_response' sends the prompt to the selected API.
# Finally, the code prints the answer of the LLM.

# After several hours of running into the issue of not being to 
# evaluate the variable names in the LLM's code part, we 
# discovered the existence of the f-strings and their necessity
# at that stage. This means that we can insert the value of a 
# variable directly into a string.

# The sources:
    # Python Software Foundation. (n.d.). input. In Built-in functions.
    # Retrieved April 1, 2025, from
    # https://docs.python.org/3/library/functions.html#input

    # Python Software Foundation. (n.d.). Formatted string literals.
    # In Lexical analysis. Retrieved April 1, 2025, from
    # https://docs.python.org/3/reference/lexical_analysis.html#f-strings

    # Python Software Foundation. (n.d.). Dictionaries. In Data structures.
    # Retrieved April 1, 2025, from
    # https://docs.python.org/3/tutorial/datastructures.html#dictionaries

    # Python Software Foundation. (n.d.). Defining functions. In The Python
    # tutorial. Retrieved April 1, 2025, from
    # https://docs.python.org/3/tutorial/controlflow.html#defining-functions


def run_from_terminal():
    prompt = input("Write your prompt: ")
    result = select_best_llm(prompt, model, vectorizer, llm_classes)
    print(f"Selected LLM: {result['selected_llm']}")
    print(f"Predicted Index (raw model output): {result['predicted_label']}")

    llm_response = call_llm_api(prompt, result['selected_llm'])
    print("\nLLM Response:\n", llm_response)


# Finally, we need to display the LLM usage chart.

# The sources:
    # Streamlit, Inc. (n.d.-b). st.bar_chart. Streamlit documentation. 
    # Retrieved May 2, 2025, from 
    # https://docs.streamlit.io/develop/api-reference/charts/st.bar_chart

    # Streamlit, Inc. (n.d.-c). Session state: Architecture. 
    # Streamlit documentation. Retrieved May 2, 2025, from 
    # https://docs.streamlit.io/develop/concepts/architecture/session-state

    # Streamlit, Inc. (n.d.-d). Chart elements. Streamlit documentation. 
    # Retrieved May 2, 2025, from 
    # https://docs.streamlit.io/develop/api-reference/charts

