# Our app project is to provide the end-user with one interface connected to
# the API of 5 LLMs (ChatGPT, Claude, Gemini, Le Chat, and Grok).

# The user writes a prompt and our regressive machine learning model will
# analyze it and choose the best-fitted LLM to answer it.

# The app will then -through the chosen LLM's API- get the answer and shows
# it to the user.

# After extensive research on how to work with the dataset, we chose to create
# a .csv file in Excel which contains 4 categories of 10 prompts each.

# Each LLM received each prompt and we graded their answer on a scale from 0
# (worst) to 10 (best). We also created a column which indicates -in the
# integer form- which LLM gave the best answer and, thus is best fitted to
# provide future answer on similar prompts.

# The integer list is as follows: ChatGPT is 0, Claude is 1, Gemini is 2,
# Le Chat is 3, and Grok is 4.

# Once the .csv file was done, we made some extensive research to find the right
# Python library to work with our tabular dataset. We found out it is the Pandas
# library.

# The source:
    # pandas development team. (n.d.). Getting started. Pandas documentation.
    # Retrieved April 1, 2025, from
    # https://pandas.pydata.org/docs/getting_started/index.html#getting-started​

import pandas as pd


# Once Pandas is imported, we need this library to interact with our .csv file.
# This can be done through Pandas' dataframe which we named -by convention- df.

# df then needs to interact with pd.read_csv to access our .csv file which is
# named "prompts.csv"

# The sources:
    # pandas development team. (n.d.). pandas.read_csv.
    # pandas documentation. Retrieved April 1, 2025, from
    # https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html

    # pandas development team. (n.d.). pandas.DataFrame. pandas documentation.
    # Retrieved April 1, 2025, from
    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html​

df = pd.read_csv("prompts.csv")


# We then ran into the issue that only the first five columns of the .csv file
# were showing. Thus, after a quick glance on internet, we found that we needed
# to use the IPython library and its display module. This would then enable us
# to use display to show the entire .csv file.

# The source:
    # IPython development team. (n.d.). IPython.display.display. IPython
    # documentation. Retrieved April 1, 2025, from
    # https://ipython.readthedocs.io/en/8.26.0/api/generated/IPython.display.html

from IPython.display import display
display(df)

from google.colab import drive
drive.mount('/content/drive')

# We know need to train our model on our data set. After some research, it
# appeared that the sklearn library would be best to train our model.

# After some extensive reading, we discovered that we needed to split our
# dataset into training and testing sets. Thus, we need to import the
# "train_test_split" module.

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
# that machine learning models can not work with text directly. Thus, we had to
# import the TF-IDF Vectorizer which formats text into vectors. This way, our
# model can understand the data it has to train on.

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


# One might think that using the RandomForestClasifier would be best. However,
# after carefully analyzing the data, we could see that each LLM responds in a
# rather predictable way to prompt specific features (e.g. coding). Thus, one
# can see linear patterns between prompt features and LLM scores.
# This part which will enable the training to happen

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


# After training our model on the data, we need a feedback loop to ensure
# that our data is qualitative enough. To do so, we relied on our Operations
# Management knowledge. We decided to use the Mean Squared Error and the R2
# score. The first tells us how wrong the model is on average, whereas the
# tells us how good the model is on average. The closer the R2 score is to 1.0,
# the more patterns the model captures.

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

df = pd.read_csv("prompts.csv")


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

# As mentionned above, plain text can not be used to train a model. Thus,
# we need to vectorize it. As we have already imported the sklearn module that
# vectorizes, we can directly vectorize it, which will add a specific and
# granular importance on each words given the context.

# The source:
    # scikit-learn developers. (n.d.). Text feature extraction. In Feature
    # extraction. scikit-learn documentation. Retrieved April 1, 2025, from
    # https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction​

vectorizer = TfidfVectorizer()


# Now, the following snippet allows us to make the model learn the vocabulary
# and transform it into a matrix. This way, it creates a matrix of the semantic
# text patterns numerically.

# The source:
    # scikit-learn developers. (n.d.).
    # sklearn.feature_extraction.text.CountVectorizer.
    # scikit-learn documentation. Retrieved April 1, 2025, from
    # https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html​

X_vect = vectorizer.fit_transform(X)

# At this stage, the data is almost fully ready to undergo the training.
# The last step is to split the data between training and testing sets.

# The code snippet is separated into the following parts:
  # X_train: it is the inputs the model will be trained on.
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
# "X = df['prompt_text']" and "y = df['best_model_label']""

# For the test_size, we used 0.2 because it allows the model to train on 95% of
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

# Now that the data is fully ready for the training and testing part, we use
# the Linear Regression to draw a line through our data and enables us to
# linearise the data. This snippet acts as the initializer for the model.

# The source:
    # scikit-learn developers. (n.d.). sklearn.linear_model.LinearRegression.
    # scikit-learn documentation. Retrieved April 1, 2025, from
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html​

model = LinearRegression()

# And now that a line has been drawn in our data, we can actually begin the
# training by making the inputs features (X_train) interact with the target
# values (y_train). It basically learns the patterns.

#The source:
    # scikit-learn developers. (n.d.). Cross-validation: evaluating estimator
    # performance. scikit-learn documentation. Retrieved April 1, 2025, from
    # https://scikit-learn.org/stable/modules/cross_validation.html​

model.fit(X_train, y_train)

# Now that the trained model is ready, we need to try the model on unseen
# test data. In this case, y_pred is a list of predicted values which the model
# thinks the correct y values are for each input in X_test.

#The source:
    # Brownlee, J. (2020, January 10). How to make predictions with
    # scikit-learn. Machine Learning Mastery. Retrieved April 1, 2025, from
    # https://machinelearningmastery.com/make-predictions-scikit-learn/​

y_pred = model.predict(X_test)


# This following two snippets are testing the quality of the learning of the
# model. Both snippets are comparing the actual values (y_test) and the
# predicted values (y_pred).

# The sources:
    # scikit-learn developers. (n.d.). Mean squared error. In Metrics and
    # scoring: quantifying the quality of predictions. scikit-learn
    # documentation. Retrieved April 1, 2025, from
    # https://scikit-learn.org/stable/modules/model_evaluation.html#mean-squared-error

    # scikit-learn developers. (n.d.). R² score. In Metrics and scoring:
    # quantifying the quality of predictions. scikit-learn documentation.
    # Retrieved April 1, 2025, from
    # https://scikit-learn.org/stable/modules/model_evaluation.html#r2-score

# After discussing it with ChatGPT, an MSE of 0.876 and a r2 score of 0.610 are
# decent results for the amount of data we have. Thus, we can continue to the next step.

print(mean_squared_error(y_test, y_pred))
print(r2_score(y_test, y_pred))

# We have completed the training stage for our app.
# Now, one needs to pick the best LLM based on each user's prompt.
# Thus, we need to use the trained model to predict a score for each LLM.

# We need to import NumPy as we will need it later in our code.

import numpy as np

# Then, we need to define the list of the LLMs that we are going to be using.
# The same that we have trained.
# The index of each model is positioned based on its label in the data set.

llm_classes = ['ChatGPT', 'Claude', 'Gemini', 'Mistral', 'Grok']

# Now, we need to define a function that will return a prompt and provide us
# back with the most adapted model to answer the user's prompt. This function
# contains 4 parameters. First, 'prompt' is the user's question. Second, 'model'
# is the trained regression model. Third, 'vectorizer' is the TF-IDF vectorizer
# used above to transform each prompt into numerical data. Finally, 'llm_classes'
# is the list of our LLM models.
# Following this function, we have 5 other dependant functions.
# The first function is 'vect_prompt' which uses the TF-IDF to transform the user's
# prompt into numerical data based on the vocabulary used during training.
# Second, 'predicted_label' uses the trained model to predict the label based on
# the prompt. This will give us a float number which represents the regressed
# estimate of the best LLM for a particular prompt.
# Third, 'best_index' is rounding the float number to the neares integer. This
# integer is the position of one of our LLM, meaning this LLM is the most fit
# to answer this particular prompt.
# Fourth, 'best_llm' uses the integer to retrieve the corresponding name of the
# best fit LLM from the 'llm_classes' list
# Lastly, 'return' provides us with all the relevant information about our
# prediction

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

def select_best_llm(prompt, model, vectorizer, llm_classes):
  vect_prompt = vectorizer.transform([prompt])
  predicted_label = model.predict(vect_prompt)[0]
  best_index = round(predicted_label)
  best_llm = llm_classes[best_index]
  return {
      'prompt': prompt,
      'predicted_label': predicted_label,
      'best_llm_index' : best_index,
      'selected_llm' : best_llm
  }

# Now we are tackling the last part of our project. We need to connect our code
# to the model's API. So, when the user's sends a prompt, our code detects which
# LLM should be called and our code sends the prompt's request to the correct
# LLMs API. Then, the LLM answers the user's prompt and sends the answer back to
# the user's screen.
# This part of the code enables us to create a logic pathway of which LLM's API
# our code needs to contact based on on the integer the user's prompt created.
# The logic is the following: if the 'selected_llm' variable is equal to 'LLM',
# then, it calls the function 'call_llm(prompt)' to send the prompt to the LLM's
# API. It centralizes the logic for sending prompts to the correct LLM.

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

# First, we need to install the LLM packages in Colab.

# OpenAI ChatGPT
import openai

# Anthropic Claude
import anthropic

# Google Gemini
import google-generativeai

# Mistral
import mistralai

# Grok
import git+https://github.com/moesmufti/xai_grok_sdk.git

# This part of the code will call OpenAI's ChatGPT API if prompted by the user's
# prompt. The code is based on OpenAI's own documentation as well as our own
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
client = openai.OpenAI(api_key = "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")
def call_chatgpt(prompt):
    try:
        response = client.chat.completions.create(
            model = "gpt-4o",
            messages = [
                {"role": "user", "content": prompt}
            ]
        )
        print(response.choices[0].message.content)
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
client = anthropic.Anthropic(api_key = "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")

def call_claude(prompt: str) -> str:
    try:
        response = client.messages.create(
            model = "claude-3-7-sonnet-20250219",
            max_tokens = 1000,
            messages = [
                {"role": "user", "content": prompt}
            ]
        )
        return response.content
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

client = Mistral(api_key = "vJQyWusvYsujsVZDZZiUjaSLDjbV8H4C")

def call_mistral(prompt):
    try:
        response = client.chat.complete(
            model = "mistral-small",
            messages = [{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Mistral API error: {e}"

# This part of the code will call XAI's API if prompted by the user's prompt.
# The code is based on OpenAI's own documentation and xAI's own documentation
# as well as our own modifications to make the code work in our workflow.

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

# Now, we want to be able to retrieve the user's prompt.
# First, we want to get the user's prompt with 'prompt'
# Second, we call our 'select_best_llm' which runs the 4 parameters.
# Finally, we display the result of the user's query as well as the selected LLM.
# This result comes from our call_llm_api which calls the correct API and brings
# the correct answer back to the user's screen.
# It is important to note that for both 'print' functions, we need to use
# an f-string to allow variable interpolation. This means that we can insert
# the value of a variable directly into a string.

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
