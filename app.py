import streamlit as st
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from peft import PeftModel
# import torch
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from spellchecker import SpellChecker
import json
import numpy as np
# from huggingface_hub import login
# import os

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer

# --- Preprocessing Components ---
STOP_WORDS = set(stopwords.words('english'))
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

# Button to predict
if st.button("Predict"):
    if text.strip():  # Check if input text is not empty
        # Preprocess user input
        seq = tokenizer.texts_to_sequences([text])  # Tokenize the input text
        padded_seq = pad_sequences(seq, maxlen=100, padding='post')  # Pad the sequence

        # Make prediction
        prediction = model.predict(padded_seq)  # Predict using the model
        predicted_class = np.argmax(prediction, axis=1)[0]  # Get the predicted class index

        # Display result
        st.write(f"Predicted Class: {predicted_class}")
    else:
        st.write("Please enter text to classify.")