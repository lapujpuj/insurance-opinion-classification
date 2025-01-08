import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# --- Charger le modèle et le tokenizer ---
@st.cache_resource
def load_model():
    # Remplacez par le chemin de votre modèle sur Hugging Face
    model_name = "pujpuj/roberta-lora-token-classification"
    base_model_name = "roberta-large"
    
    # Charger le modèle et le tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model()

# --- Interface Streamlit ---
st.title("Prédiction de Note")
st.write("Entrez un avis ci-dessous pour prédire sa note.")

# Entrée utilisateur
text = st.text_area("Avis :", "")

# Bouton pour prédire
if st.button("Prédire"):
    if text.strip() != "":
        # Tokenisation
        inputs = tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Prédiction
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_label = torch.argmax(logits, dim=1).item()

        # Afficher le résultat
        st.write(f"**Note prédite : {predicted_label}**")
    else:
        st.write("Veuillez entrer un avis.")

