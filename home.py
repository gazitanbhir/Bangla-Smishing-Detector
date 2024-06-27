# app.py

import streamlit as st
from utils import predict_text

st.title("Bangla SMISH Text Classification")

text = st.text_area("Enter Your Bangla Text")

if st.button("Classify"):
    if text.strip() != "":
        predicted_class, label_encoder = predict_text(text)
        predicted_label = label_encoder.inverse_transform([predicted_class])
        st.success(f"Predicted Label: {predicted_label[0]}")
    else:
        st.write("Please enter some text for classification.")
