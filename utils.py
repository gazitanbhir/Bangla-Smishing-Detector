# utils.py

import re
import torch
from transformers import BertForSequenceClassification, BertTokenizer
import joblib

# Load the pre-trained model and tokenizer
model = BertForSequenceClassification.from_pretrained('/home/hydra/Documents/Streamlit/smish/model')
tokenizer = BertTokenizer.from_pretrained('/home/hydra/Documents/Streamlit/smish/model')

# Load the label encoder
label_encoder = joblib.load('/home/hydra/Documents/Streamlit/smish/label_encoder.pkl')

# Set the model to evaluation mode
model.eval()

# Set the device to GPU or CPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Function to clean text
def clean_text(text):
    # Remove emojis and special characters (keeping Bangla characters, English characters, and numbers)
    text = re.sub(r'[^\w\s\u0980-\u09FF]', '', text)
    # Remove extra spaces
    text = ' '.join(text.split())
    return text

# Function to preprocess and make predictions
def predict_text(text):
    # Clean the text
    cleaned_text = clean_text(text)
    
    # Tokenize the cleaned text
    inputs = tokenizer(cleaned_text, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
    
    # Move input tensors to the device (GPU or CPU)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    # Make predictions
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
    
    # Get the predicted class
    predicted_class = torch.argmax(logits, dim=-1).item()
    
    return predicted_class, label_encoder
