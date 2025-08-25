import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib

# Load pre-trained models and vectorizers (excluding SVM for now)
@st.cache_resource
def load_resources():
    try:
        tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
        logistic_model = joblib.load("logistic_regression_model.pkl")
        bilstm_model = load_model("bilstm_model.h5")
        return tfidf_vectorizer, logistic_model, bilstm_model
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        st.error("Please run the notebook first to train and save the models!")
        return None, None, None

# Clear cache button
if st.button("Clear Cache and Reload Models"):
    st.cache_resource.clear()
    st.rerun()

# Check if models are available
models_loaded = False
try:
    tfidf_vectorizer, logistic_model, bilstm_model = load_resources()
    if all(model is not None for model in [tfidf_vectorizer, logistic_model, bilstm_model]):
        models_loaded = True
except:
    models_loaded = False

# Define preprocessing function
def preprocess_text(text):
    # Add your text preprocessing steps here
    return text.lower()

# Streamlit app
st.title("AI Text Detector")

if not models_loaded:
    st.error("⚠️ Model files are missing!")
    
    # Check which files are missing
    required_files = [
        "tfidf_vectorizer.pkl",
        "logistic_regression_model.pkl", 
        "bilstm_model.h5"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        st.write("Missing files:")
        for file in missing_files:
            st.write(f"❌ {file}")
    
    st.info("Please run the notebook first to train and save the models:")
    st.code("""
    1. Open ai_text_detector_new.ipynb
    2. Run all cells from top to bottom
    3. Wait for all models to be trained and saved
    4. Then refresh this Streamlit app
    """)
    st.stop()

st.write("This app detects whether a given text is AI-generated or human-written using various models.")

# Dropdown menu for model selection (excluding SVM for now)
model_choice = st.selectbox("Select a model to use:", [
    "Logistic Regression",
    "BiLSTM",
    "BERT"
])

# Display predictions from the corresponding CSV file
if st.button("Show Predictions"):
    file_mapping = {
        "Logistic Regression": "logistic_regression_predictions.csv",
        "BiLSTM": "bilstm_predictions.csv",
        "BERT": "bert_predictions.csv"
    }

    selected_file = file_mapping.get(model_choice)
    try:
        predictions_df = pd.read_csv(selected_file)
        st.write(predictions_df)
    except FileNotFoundError:
        st.error(f"Predictions file for {model_choice} not found. Please ensure the model has been trained.")

# Input text for testing
st.subheader("Test New Data")
user_input = st.text_area("Enter text to analyze:")

if st.button("Analyze"):
    if user_input.strip():
        # Preprocess input
        processed_text = preprocess_text(user_input)

        if model_choice == "Logistic Regression":
            # TF-IDF transformation
            tfidf_features = tfidf_vectorizer.transform([processed_text])
            # Get probability scores instead of just prediction
            prediction_proba = logistic_model.predict_proba(tfidf_features)[0]
            human_prob = prediction_proba[0] * 100  # Class 0 = Human
            ai_prob = prediction_proba[1] * 100     # Class 1 = AI
            
        elif model_choice == "BiLSTM":
            # Tokenize and pad sequences
            tokenizer = joblib.load("bilstm_tokenizer.pkl")  # Updated filename
            max_len = 100
            sequences = tokenizer.texts_to_sequences([processed_text])
            padded_sequences = pad_sequences(sequences, maxlen=max_len)
            # Get probability instead of binary prediction
            ai_prob_raw = bilstm_model.predict(padded_sequences)[0][0]
            ai_prob = float(ai_prob_raw) * 100
            human_prob = float(1 - ai_prob_raw) * 100
            
        elif model_choice == "BERT":
            # Load BERT tokenizer and model
            bert_tokenizer = joblib.load("bert_tokenizer.pkl")
            bert_model = load_model("bert_model.h5")
            
            # Encode text for BERT
            encoded = bert_tokenizer.encode(
                processed_text,
                add_special_tokens=True,
                max_length=64,
                padding='max_length',
                truncation=True,
            )
            input_ids = np.array([encoded])
            
            # Get probability
            ai_prob_raw = bert_model.predict(input_ids)[0][0]
            ai_prob = float(ai_prob_raw) * 100
            human_prob = float(1 - ai_prob_raw) * 100

        # Display results with percentages
        st.subheader("📊 Prediction Results")
        
        # Create columns for better layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("🤖 AI-Generated", f"{ai_prob:.1f}%")
        with col2:
            st.metric("👤 Human-Written", f"{human_prob:.1f}%")
        
        # Determine final prediction
        if ai_prob > human_prob:
            st.success(f"**Final Prediction: AI-Generated** (Confidence: {ai_prob:.1f}%)")
        else:
            st.info(f"**Final Prediction: Human-Written** (Confidence: {human_prob:.1f}%)")
        
        # Add a progress bar for visual representation
        st.subheader("📈 Confidence Visualization")
        st.progress(float(ai_prob) / 100)
        st.caption(f"AI Likelihood: {ai_prob:.1f}% | Human Likelihood: {human_prob:.1f}%")
    else:
        st.warning("Please enter some text to analyze.")
