import streamlit as st
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from peft import PeftModel
# import torch
import re
import subprocess
import shap
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from pyngrok import ngrok, conf
from spellchecker import SpellChecker
import time
import json
import numpy as np
from streamlit.components.v1 import iframe
import os
import matplotlib.pyplot as plt
import random
# from huggingface_hub import login
# import os


# # by the way might imply add library to requirement
# streamlit
# # torch
# # transformers
# # peft
# tensorflow
# gensim
# pyspellchecker
# nltk
# json
# numpy
# # huggingface_hub

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)


# from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer

# --- Preprocessing Components ---
STOP_WORDS = set(nltk.corpus.stopwords.words('english'))
TOKENIZER = TreebankWordTokenizer()
REGEX = re.compile(r'http\S+|www\S+|https\S+|<[^>]+>')  # To remove URLs and HTML tags
EMOJI_PATTERN = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags
                           "]+", flags=re.UNICODE)
LEMMATIZER = WordNetLemmatizer()
SPELL = SpellChecker()
REGEX_NON_ALPHANUM = re.compile(r'[^a-zA-Z\s]')  # Remove non-alphabetic characters

# Function to expand contractions
def expand_contractions(text):
    contractions_dict = {
        "can't": "cannot",
        "won't": "will not",
        "n't": " not",
        "'re": " are",
        "'s": " is",
        "'d": " would",
        "'ll": " will",
        "'t": " not",
        "'ve": " have",
        "'m": " am"
    }
    pattern = re.compile('({})'.format('|'.join(contractions_dict.keys())), flags=re.IGNORECASE | re.DOTALL)

    def replace(match):
        return contractions_dict.get(match.group(0).lower(), match.group(0))

    return pattern.sub(replace, text)

# Function to correct spelling
def correct_spelling(tokens):
    corrected_tokens = []
    misspelled = SPELL.unknown(tokens)
    for word in tokens:
        if word in misspelled:
            corrected = SPELL.correction(word)
            if corrected:
                corrected_tokens.append(corrected)
            else:
                corrected_tokens.append(word)
        else:
            corrected_tokens.append(word)
    return corrected_tokens

# Function to clean and tokenize text
def clean_and_tokenize(text):
    text = REGEX.sub('', text)  # Remove URLs and HTML tags
    text = EMOJI_PATTERN.sub('', text)  # Remove emojis
    text = expand_contractions(text)  # Expand contractions
    text = REGEX_NON_ALPHANUM.sub(' ', text)  # Remove non-alphabetic characters
    text = text.lower()  # Convert to lowercase
    tokens = TOKENIZER.tokenize(text)  # Tokenize
    tokens = [word for word in tokens if word not in STOP_WORDS and len(word) > 1]  # Remove stop words
    tokens = correct_spelling(tokens)  # Correct spelling
    tokens = [LEMMATIZER.lemmatize(word) for word in tokens]  # Lemmatization
    return tokens

# --- Load model and tokenizer for Roberta ---
# @st.cache_resource
# def load_model():
#     base_model_name = "roberta-large"  # Base model
#     adapter_model_name = "pujpuj/roberta-lora-token-classification"  # Your repo with LoRA

#     tokenizer = AutoTokenizer.from_pretrained(base_model_name)  # Load tokenizer
#     base_model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=5)  # Load base model
#     model = PeftModel.from_pretrained(base_model, adapter_model_name)  # Load LoRA adapters
#     return model, tokenizer


# # --- Load model and tokenizer for Llama 3.2 1B ---
# # Authenticate with Hugging Face
# token = os.getenv("HF_TOKEN")  # Fetch the token from the environment
# login(token=token)


# @st.cache_resource
# def load_model():
#     # Updated base model and adapter model
#     base_model_name = "meta-llama/Llama-3.2-1B"  # Base LLaMA model
#     adapter_model_name = "ahmedmaaloul/insurance-opinion-classification-llama-3.2-1b-LoRa"  # Hugging Face repo

#     # Load tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(base_model_name, add_prefix_space=True)
#     tokenizer.pad_token_id = tokenizer.eos_token_id
#     tokenizer.pad_token = tokenizer.eos_token

#     # Load base model
#     base_model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=5)

#     # Set the pad token id on the base model config
#     base_model.config.pad_token_id = tokenizer.pad_token_id


#     # Load LoRA adapter from Hugging Face Hub
#     model = PeftModel.from_pretrained(base_model, adapter_model_name)

#      return model, tokenizer

# model, tokenizer = load_model()


@st.cache_resource
def load_model_and_tokenizer():
    """
    Load the trained TensorFlow model and tokenizer.
    """
    # Load the saved model
    model = load_model('saved_model/Basic_Embedding_NN.h5')
    
    # Load the tokenizer
    with open('saved_model/tokenizer.json') as f:
        tokenizer_json = json.load(f)
    tokenizer = tokenizer_from_json(tokenizer_json)
    
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# --- Streamlit Interface ---
st.title("Prediction of Insurance Review Rating")
st.write("Enter an insurance review below to predict its rating.")

# User input
text = st.text_area("Review:", "")

# # Button for prediction
# if st.button("Predict"):
#     if text.strip() != "":
#         # Preprocess the input
#         tokens = clean_and_tokenize(text)
#         cleaned_text = " ".join(tokens)  # Reconstruct the cleaned text
        
#         # Tokenization
#         inputs = tokenizer(
#             cleaned_text,
#             padding=True,
#             truncation=True,
#             max_length=512,
#             return_tensors="pt"
#         )
        
#         # Prediction
#         model.eval()
#         with torch.no_grad():
#             outputs = model(**inputs)
#             logits = outputs.logits
#             predicted_label = torch.argmax(logits, dim=1).item() + 1

#         # Display result
#         st.write(f"**Cleaned Review:** {cleaned_text}")
#         st.write(f"**Predicted Rating:** {predicted_label}")
#     else:
#         st.write("Please enter a review.")

# Répertoire contenant les fichiers TensorBoard

if st.button("Predict"):

    if text.strip():
        text_seq = tokenizer.texts_to_sequences([text])  # Tokenisation
        text_padded = pad_sequences(text_seq, maxlen=100, padding='post')  # Padding

        # Prédiction
        prediction = model.predict(text_padded)
        predicted_class = np.argmax(prediction)  # Classe prédite

        # Affichage des résultats
        st.subheader("Prediction")
        st.write(f"**Predicted Class:** {predicted_class}")
        # st.write(f"**Prediction Probabilities:** {prediction}")

        # --- Analyse SHAP ---
        st.subheader("SHAP Analysis")

        # Fonction de prédiction adaptée pour SHAP
        def predict_proba(padded_texts):
            """
            Fonction pour générer des prédictions basées sur le modèle.
            """
            return model.predict(padded_texts)  # Retourne des probabilités (shape: [n_samples, n_classes])

        # Création de l'explainer SHAP avec KernelExplainer
        explainer = shap.KernelExplainer(
            predict_proba,
            np.zeros((1, text_padded.shape[1]))  # Une séquence de base remplie de zéros
        )

        # Calcul des valeurs SHAP
        shap_values = explainer.shap_values(text_padded)

        # Extraction des mots importants (ignorer "<OOV>")
        sequence_words = [tokenizer.index_word.get(idx, None) for idx in text_seq[0] if idx != 0]
        shap_importances = shap_values[0][0][:len(sequence_words)]  # Valeurs SHAP pour les mots
        word_importance_pairs = list(zip(sequence_words, shap_importances))

        # Filtrer pour exclure les mots "None" (correspondant à <OOV>)
        filtered_pairs = [(word, importance) for word, importance in word_importance_pairs if word is not None]

        # Trie les mots par importance
        sorted_importance = sorted(filtered_pairs, key=lambda x: abs(x[1]), reverse=True)


        # Préparer les données pour le graphique SHAP
        words, importances = zip(*filtered_pairs)
        shap_matrix = np.array([importances])  # Ajouter une dimension pour représenter un seul exemple

        # Visualisation des importances avec SHAP bar_plot
        plt.figure(figsize=(8, 6))
        shap.summary_plot(
            shap_matrix,  # Matrice des valeurs SHAP (1, n_features)
            feature_names=words,  # Les mots correspondants
            plot_type="bar",  # Type de graphique : bar
            show=False
        )
        plt.title("SHAP Feature Importance")
        st.pyplot(plt)  # Intègre le graphique dans Streamlit
            

        

    else:
        st.write("Please enter a review.")

# Affichage de TensorBoard

projector_log_dir = "projector"

ngrok_token = os.getenv("NGROK_TOKEN")


# Configurer ngrok avec la clé
conf.get_default().auth_token = ngrok_token

# Répertoire contenant les fichiers TensorBoard
projector_log_dir = "projector"

if not os.path.exists(projector_log_dir):
    st.error(f"The directory '{projector_log_dir}' does not exist. Please check TensorBoard setup.")
else:

    # Lancer TensorBoard
    tensorboard_command = f"tensorboard --logdir projector --host=0.0.0.0 --port=6010"
    subprocess.Popen(tensorboard_command, shell=True)

    # Attendre quelques secondes pour que TensorBoard soit prêt
    time.sleep(5)

    # Créer un tunnel ngrok pour exposer TensorBoard
    public_url = ngrok.connect(6010, "http")
    st.success(f"TensorBoard public URL: {public_url}")

    # Intégrer TensorBoard dans Streamlit
    st.subheader("Embedding Visualization via TensorBoard")
    iframe(public_url, height=800, scrolling=True)