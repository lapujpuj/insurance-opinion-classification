import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import torch

# --- Load model and tokenizer ---
@st.cache_resource
def load_model():
    # Load the base model from Hugging Face
    base_model_name = "roberta-large"  # Base model
    adapter_model_name = "pujpuj/roberta-lora-token-classification"  # Your repo with LoRA

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # Load the base model
    base_model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=5)

    # Apply LoRA adapters to the base model
    model = PeftModel.from_pretrained(base_model, adapter_model_name)

    return model, tokenizer


model, tokenizer = load_model()

# --- Streamlit Interface ---
st.title("Prediction of Rating")
st.write("Enter a review below to predict its rating.")

# User input
text = st.text_area("Review:", "")

# Button for prediction
if st.button("Predict"):
    if text.strip() != "":
        # Tokenization
        inputs = tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Prediction
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_label = torch.argmax(logits, dim=1).item() + 1

        # Display result
        st.write(f"**Predicted rating: {predicted_label}**")
    else:
        st.write("Please enter a review.")
