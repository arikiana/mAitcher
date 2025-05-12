# ChatGPT wurde zur Korrektur, dem de-bugging und zur Anleitung für den folgenden Codes verwendet.
# die Quellen zu den einzelnen Schritten werden jeweils unterhalb des jeweiligen Codes aufgeführt, wobei die Erklärungen entweder daneben oder darüber stehen.


# Unser Projekt ist eine Ein-Platform-Lösung, welche User mit den APIs von 5 LLMs (ChatGPT, Claude, Gemini, Le Chat, and Grok) verbindet.
# Die Benutzer schreiben einen Prompt und unser Machine-learning-modell analysiert diesen und weist ihn mittels einer linearen Regression dem best passenden LLM zu.
# Die App wird dann über die API des gewählten AI-Modells dem User eine Antwort zurückgeben.


# Nach langer Recherche zur Benutzung von Datasets haben wir uns entschieden ein .csv File in Excel zu erstellen, welches 4 Kategorien mit 10 Prompts enthält.
# Jedem LLM wurden alle Prompts gestellt und die entsprechenden Antworten auf einer Skala von 0(schlecht) bis 10 (sehr gut) bewertet.  
# Eine weitere Spalte zur Erfassung des best-antwortenden AI-Modells je Kategorie wird als Anhaltspunkt verwendet, dass dieses LLM am besten zur Beantwortung derartiger Prompts geeignet ist.

# Die LLMs sind wie folgt aufgelistet: ChatGPT ist 0, Claude ist 1, Gemini ist 2, Le Chat ist 3, and Grok ist 4.



import streamlit as st # Um die App auf Streamlit bauen zu können und darüber zu deployen.
import pandas as pd # Die Pandas Library erwies sich am geeignetsten, um mit unserem tabellenförmigen Dataset zu arbeiten.
import numpy as np # Wird ebenfalls für den Code benötigt.

# Quellen:
    # pandas development team. (o.D.). Getting started. Pandas documentation. Abgerufen am 1 April, 2025, von https://pandas.pydata.org/docs/getting_started/index.html#getting-started​


df = pd.read_csv("prompts.csv") # Um mit unserem .csv File (Namens "prompts.csv) zu interagieren, wird über die Pandas Dataframe (pd) Funktion pd.read_csv auf das .csv zugegriffen und in ein df eingelesen.

 # Quellen:
    # pandas development team. (o.D.). pandas.read_csv. pandas documentation. Abgerufen am 1. April, 2025, von https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
    # IPython development team. (o.D.). IPython.display.display. IPython documentation. Abgerufen am 1. April, 2025, von https://ipython.readthedocs.io/en/8.26.0/api/generated/IPython.display.html
    # pandas development team. (o.D.). pandas.DataFrame. pandas documentation. Abgerufen am 1. April, 2025, von https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html​


# Um unser Modell auf unserem Dataset zu trainieren, haben wir uns nach einiger Recherche für die sklearn Library entschieden.

from sklearn.model_selection import train_test_split # Unser Dataset wird mittels "train_test_split" Modul in Training- und Testing-sets aufgeteilt.

# Quellen:
    # Google Developers. (o.D.). Dividing the original dataset. In Machine Learning Crash Course. Abgerufen am 1. April, 2025, von https://developers.google.com/machine-learning/crash-course/overfitting/dividing-datasets​
    # scikit-learn developers. (o.D.). Computing cross-validated metrics. scikit-learn documentation. Abgerufen am 1. April, 2025, von https://scikit-learn.org/stable/modules/cross_validation.html#computing-cross-validated-metrics​
    # scikit-learn developers. (o.D.). sklearn.model_selection.train_test_split. scikit-learn documentation. Abgerufen am 1. April, 2025, von https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html​


# Da Machine Learning nicht in der Art mit Text umgehen kann, in welcher wir beabsichtigten, mussten wir den TF-idf Vectorizer importieren. 

from sklearn.feature_extraction.text import TfidfVectorizer # Wandelt den Text in numerische Vektoren um, damit unser Modell die Daten, mit welchen es trainiert, versteht. 

# Quellen:
    # scikit-learn developers. (o.D.). Text feature extraction. scikit-learn documentation. Abgerufen am 1. April, 2025, von https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction
    # scikit-learn developers. (o.D.). sklearn.feature_extraction.text.TfidfVectorizer. scikit-learn documentation. Abgerufen am 1. April, 2025, von https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    # Masudowolabi. (2024, 10. Juni). How to use sklearn’s TfidfVectorizer for text feature extraction in model testing. Medium. Abgerufen am 1. April, 2025, von https://medium.com/@masudowolabi/how-to-use-sklearns-tfidfvectorizer-for-text-feature-extraction-in-model-testing-e1221fd274f8


# In der Analyze der Daten wurde ersichtlich, dass jedes LLM sehr vorhersehbar auf Prompts mit spezifischen Elementen wie "coding" antwortet. Dadurch wurden lineare Muster zwischen Prompt-Elementen und LLM scores ersichtlich.
# Eine LinearRegression aus der sklearn Library erschien daher als nützliches Tool.

from sklearn.linear_model import LinearRegression

# Quellen:
    # scikit-learn developers. (o.D.). Linear models. scikit-learn documentation. Abgerufen am 1. April, 2025, von https://scikit-learn.org/stable/modules/linear_model.html​
    # scikit-learn developers. (o.D.). sklearn.ensemble.RandomForestClassifier. scikit-learn documentation. Abgerufen am 1. April, 2025, von https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html​
    # scikit-learn developers. (o.D.). sklearn.linear_model.LinearRegression. scikit-learn documentation. Abgerufen am 1. April, 2025, von https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html


# Ein Feedback Loop zur Bewertung der Qualität unserer Daten wurde mittels Mean Squared Error (MSE = durchschnittlicher Fehler zwischen dem predicted und dem actial value) und R2 (wie sehr erklärt unser Modell die Variabilität in der Zielvariable) erstellt.
# Je niedriger der MSE und je näher das R2 an 1.0, desto besser ist das Modell.

from sklearn.metrics import mean_squared_error, r2_score

#Quellen:
    # scikit-learn developers. (o.D.). sklearn.metrics.mean_squared_error. scikit-learn documentation. Abgerufen am 1. April, 2025, von https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
    # scikit-learn developers. (o.D.). Mean squared error. In Metrics and scoring: Quantifying the quality of predictions. scikit-learn documentation. Abgerufen am 1. April, 2025, von https://scikit-learn.org/stable/modules/model_evaluation.html#mean-squared-error
    # scikit-learn developers. (o.D.). sklearn.metrics.r2_score. scikit-learn documentation. Abgerufen am 1. April, 2025, von https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html
    # scikit-learn developers. (o.D.). R² score. In Metrics and scoring: Quantifying the quality of predictions. scikit-learn documentation. Abgerufen am 1. April, 2025, von https://scikit-learn.org/stable/modules/model_evaluation.html#r2-score


X = df['prompt_text'] # Extrahiert die Input Features, mit denen das Modell lernt.
y = df['best_model_label'] # Extrahiert die Zielvariablen, also was das Modell vorherzusagen Versucht. Die Zielvariable ist das Modell, welches den höchsten Score für einen spezifischen Prompt hatte.

# Quellen:
    # scikit-learn developers. (o.D.). Feature selection. scikit-learn documentation. Abgerufen am 1. April, 2025, von https://scikit-learn.org/stable/modules/feature_selection.html​
    # pandas development team. (o.D.). pandas.DataFrame. pandas documentation (version 0.23). Abgerufen am 1. April, 2025, von https://pandas.pydata.org/pandas-docs/version/0.23/generated/pandas.DataFrame.html​


vectorizer = TfidfVectorizer() # Verschafft jedem Wort Wichtigkeit je nach Kontext (siehe oben für Begründung).

# Quellen:
    # scikit-learn developers. (o.D.). Text feature extraction. In Feature extraction. scikit-learn documentation. Abgerufen am 1. April, 2025, von https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction​


X_vect = vectorizer.fit_transform(X) # Um dem Modell zu ermöglichen Vokabular zu lernen und eine Matrix zu erstellen. Erstellt eine numerische Matrix der semantischen Textmuster.

# Quellen:
    # scikit-learn developers. (o.D.). sklearn.feature_extraction.text.CountVectorizer. scikit-learn documentation. Abgerufen am 1. April, 2025, von https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html​


X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size = 0.05, random_state = 21) # Unterteilt die Daten in Trainings- und Test-sets, wobei die test_size 0.05 gewählt wurde, um mit 95% der Daten zu lernen, da nicht viele Daten zur Verfügung stehen.
                                                                                                    # Daher möchten wir die Trainigs-iterationen möglichst erhöhen.
# Quellen:
    # scikit-learn developers. (o.D.). sklearn.model_selection.train_test_split. scikit-learn documentation. Abgerufen am 1. April, 2025, von https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html​


model = LinearRegression() # Zieht eine Trendlinie durch die Daten für Vorhersagen (Begründung zur Wahl siehe oben).

# Quellen:
    # scikit-learn developers. (o.D.). sklearn.linear_model.LinearRegression. scikit-learn documentation. Abgerufen am 1. April, 2025, von https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html​


model.fit(X_train, y_train) # Bringt dem Modell Muster bei, indem die Input-Features mit den Target-Values interagieren.

#Quellen:
    # scikit-learn developers. (o.D.). Cross-validation: evaluating estimator performance. scikit-learn documentation. Abgerufen am 1. April, 2025, von https://scikit-learn.org/stable/modules/cross_validation.html​


y_pred = model.predict(X_test) # Um die Validität der Vorhersagen des Modells zu testen.

#Quellen:
    # Brownlee, J. (2020, 10 Januar). How to make predictions with scikit-learn. Machine Learning Mastery. Abgerufen am 1. April, 2025, von https://machinelearningmastery.com/make-predictions-scikit-learn/​


print(mean_squared_error(y_test, y_pred)) # Gibt den MSE als Feedback für das Modell zurück.
print(r2_score(y_test, y_pred)) # Gibt R2 als Feedback für das Modell zurück.

# Quellen:
    # scikit-learn developers. (o.D.). Mean squared error. In Metrics and scoring: quantifying the quality of predictions. scikit-learn documentation. Abgerufen am 1. April, 2025, von https://scikit-learn.org/stable/modules/model_evaluation.html#mean-squared-error
    # scikit-learn developers. (o.D.). R² score. In Metrics and scoring: quantifying the quality of predictions. scikit-learn documentation. Abgerufen am 1. April, 2025, von https://scikit-learn.org/stable/modules/model_evaluation.html#r2-score

# Nach einer Unterhaltung mit ChatGPT, entschieden wir, dass ein MSE von 0.876 und ein R2 von 0.610 genügend sind, gegeben der Menge an zur Verfügung stehenden Daten.
# Das Modell wird nun also benutzt, um einen Score für jedes LLM zu berechnen.


llm_classes = ['ChatGPT', 'Claude', 'Gemini', 'Mistral', 'Grok'] # Der Index der LLMs in der Liste entspricht den Labels in unserem Daten-set.


# Funktion zur Auswahl des LLM:
def select_best_llm(prompt): # Gibt dem User-prompt die Antwort des besten LLM zurück.
    vect_prompt    = vectorizer.transform([prompt]) # Verwendet TF-idf, um den User-prompt in gleichartige Vektoren zu verwandeln, welche auch beim Training benutzt wurden.
    predicted_label = model.predict(vect_prompt)[0] # Mittels des trainierten Modells wird ein Float zwischen 0 & 5 ausgegeben. Dieser korrespondiert mit den 5 LLMs aus unserer Liste.
    best_index      = round(predicted_label) # Rundet den Float zum nächsten Integer auf/ab. Dieses LLM wird dann zur Antwort verwendet.
    selected_llm    = llm_classes[best_index] # Holt das beste LLM, welches zu dem gerundeten Integer gehört aus der Liste der LLMs.
    return selected_llm, predicted_label # Gibt alle relevanten Informationen zurück.

    #Funktion zur Dokumentation des Gebrauchs der LLMs:
    if 'llm_usage' not in st.session_state: 
        st.session_state.llm_usage = {llm: 0 for llm in llm_classes} 
    st.session_state.llm_usage[choice] += 1 # Fügt neue Daten der LLM-Auswahl dem Chart hinzu.
    return choice, pred 

# Quellen:
    # Python Software Foundation. (o.D.). More on lists. In The Python tutorial. Abgerufen am 1. April, 2025, von https://docs.python.org/3/tutorial/datastructures.html#more-on-lists​
    # NumPy developers. (o.D.). What is NumPy?. NumPy documentation. Abgerufen am 1. April, 2025, von https://numpy.org/devdocs/user/whatisnumpy.html​
    # Python Software Foundation. (o.D.). Defining functions. In The Python tutorial. Abgerufen am 1. April, 2025, von https://docs.python.org/3/tutorial/controlflow.html#defining-functions
    # scikit-learn developers. (o.D.). sklearn.feature_extraction.text.TfidfVectorizer. scikit-learn documentation. Abgerufen am 1. April, 2025, von https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    # scikit-learn developers. (o.D.). sklearn.linear_model.LinearRegression.predict. scikit-learn documentation. Abgerufen am 1. April, 2025, von https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression.predict
    # Python Software Foundation. (o.D.). round. In Built-in functions. Abgerufen am 1. April, 2025, von https://docs.python.org/3/library/functions.html#round
    # Python Software Foundation. (o.D.). Lists. In An informal introduction to Python. Abgerufen am 1. April, 2025, von https://docs.python.org/3/tutorial/introduction.html#lists
    # Python Software Foundation. (o.D.). Dictionaries. In Data structures. Abgerufen am 1. April, 2025, von https://docs.python.org/3/tutorial/datastructures.html#dictionaries
    # Streamlit, Inc. (o.D.-a). st.session_state. Streamlit documentation. Abgerufen am 2. Mai, 2025, von https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state
    # Streamlit, Inc. (o.D.-b). Session state: Architecture. Streamlit documentation. Abgerufen am 3. Mai, 2025, von https://docs.streamlit.io/develop/concepts/architecture/session-state
    # Python Software Foundation. (o.D.). Dictionaries. In The Python Tutorial. Abgerufen am 3. Mai, 2025, von https://docs.python.org/3/tutorial/datastructures.html#dictionaries
    # Python Software Foundation. (o.D.). The return statement. In The Python Reference. Abgerufen am 3. Mai, 2025, von https://docs.python.org/3/reference/simple_stmts.html#the-return-statement


# Grund-Funktion, um den Prompt an das richtige LLM zu senden:
def call_llm_api(prompt, selected_llm): # Sendet den Prompt an die API des ausgewählten AI-Modells und gibt die Antwort auf den Screen des Users zurück.
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


# Funktion zum Abruf der API von OpenAI. Hierfür wurde die Dokumentation von OpenAI verwendet und angepasst:
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

# Quellen:
    # OpenAI. (o.D.). OpenAI Python library. OpenAI API documentation. Abgerufen am 1. April, 2025, von https://platform.openai.com/docs/libraries/python-library?language=python
    # OpenAI. (o.D.). Step 2: Set up your API key. OpenAI API documentation. Abgerufen am 1. April, 2025, von https://platform.openai.com/docs/quickstart/step-2-set-up-your-api-key?api-mode=responses
    # Python Software Foundation. (o.D.). Defining functions. In The Python tutorial. Abgerufen am 1. April, 2025, von https://docs.python.org/3/tutorial/controlflow.html#defining-functions


# Funktion zum Abruf der API von Claude. Hierfür wurde die Dokumentation von Anthropic verwendet und angepasst:
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

# Quellen:
    # Anthropic. (o.D.). Anthropic documentation. Abgerufen am 1. April, 2025, von https://docs.anthropic.com/en/home
    # Anthropic. (o.D.). Messages API. Abgerufen am 1. April, 2025, von https://docs.anthropic.com/en/api/messages
    # Anthropic. (o.D.). Getting started. Abgerufen am 1. April, 2025, von https://docs.anthropic.com/en/api/getting-started
    # Anthropic. (o.D.). Initial setup. Abgerufen am 1. April, 2025, von https://docs.anthropic.com/en/docs/initial-setup


# Funktion zum Abruf der API von Gemini. Hierfür wurde die Dokumentation von Google verwendet und angepasst:
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

# Quellen:
    # Google. (o.D.). Gemini API documentation. Abgerufen am 1. April, 2025, von from https://ai.google.dev/gemini-api/docs
    # Google. (o.D.). Gemini API quickstart. Abgerufen am 1. April, 2025, von from https://ai.google.dev/gemini-api/docs/quickstart
    # Google. (o.D.). Gemini models. Abgerufen am 1. April, 2025, von from https://ai.google.dev/gemini-api/docs/models
    # Google. (o.D.). Gemini API available regions. Abgerufen am 1. April, 2025, von https://ai.google.dev/gemini-api/docs/available-regions



# Funktion zum Abruf der API von Mistral. Hierfür wurde die Dokumentation von Mistral verwendet und angepasst:
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

# Quellen:
    # Mistral AI. (o.D.). Mistral AI documentation. Abgerufen am 1. April, 2025, von https://docs.mistral.ai/
    # Mistral AI. (o.D.). Mistral AI API. Abgerufen am 1. April, 2025, von https://docs.mistral.ai/api/


# Funktion zum Abruf der API von XAI. Hierfür wurde die Dokumentation von OpenAI und xAI verwendet und angepasst. xAIs API braucht die Struktur von OpenAI um in Python verwendet werden zu können:
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

# Quellen:
    # moesmufti. (o.D.). xai_grok_sdk [Computer software]. GitHub. Abgerufen am 1. April, 2025, von https://github.com/moesmufti/xai_grok_sdk


# Funktion, um die Resultate der Konversation von mAItcher und User darzustellen:
def run_from_terminal():
    prompt = input("Write your prompt: ") # Um den Prompt des Users zu erhalten.
    result = select_best_llm(prompt, model, vectorizer, llm_classes) # Gibt das ausgewählte LLM an.
    print(f"Selected LLM: {result['selected_llm']}")
    print(f"Predicted Index (raw model output): {result['predicted_label']}")

    llm_response = call_llm_api(prompt, result['selected_llm']) # Gibt der API den Prompt und holt deren Antwort.
    print("\nLLM Response:\n", llm_response)

# Quellen:
    # Python Software Foundation. (o.D.). input. In Built-in functions. Abgerufen am 1. April, 2025, von https://docs.python.org/3/library/functions.html#input
    # Python Software Foundation. (o.D.). Formatted string literals. In Lexical analysis. Abgerufen am 1. April, 2025, von https://docs.python.org/3/reference/lexical_analysis.html#f-strings
    # Python Software Foundation. (o.D.). Dictionaries. In Data structures. Abgerufen am 1. April, 2025, von https://docs.python.org/3/tutorial/datastructures.html#dictionaries
    # Python Software Foundation. (o.D.). Defining functions. In The Python tutorial. Abgerufen am 1. April, 2025, von https://docs.python.org/3/tutorial/controlflow.html#defining-functions
