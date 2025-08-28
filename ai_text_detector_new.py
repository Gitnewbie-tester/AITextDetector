import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import pickle
from transformers import pipeline
import torch

# Set page config
st.set_page_config(
    page_title="Advanced AI Text Detector",
    page_icon="🤖",
    layout="wide"
)

# Load pre-trained Hugging Face models
@st.cache_resource
def initialize_pipelines():
    """Initialize the AI detection pipelines"""
    pipelines = {}
    loading_status = {}
    
    models_to_load = {
        'chatgpt': 'Hello-SimpleAI/chatgpt-detector-roberta',
        'openai': 'openai-community/roberta-base-openai-detector'
    }
    
    for key, model_id in models_to_load.items():
        try:
            st.write(f"Loading {model_id}...")
            pipelines[key] = pipeline(
                "text-classification", 
                model=model_id,
                max_length=512,
                truncation=True,
                device=0 if torch.cuda.is_available() else -1
            )
            loading_status[key] = "✅ Loaded"
        except Exception as e:
            st.error(f"❌ Failed to load {model_id}: {e}")
            loading_status[key] = f"❌ Error: {str(e)[:50]}..."
    
    return pipelines, loading_status

def predict_with_pipeline(pipeline_model, text):
    """Make prediction with a Hugging Face pipeline"""
    try:
        result = pipeline_model(text)
        if isinstance(result, list):
            result = result[0]
        
        label = result.get('label', 'UNKNOWN')
        score = result.get('score', 0.5)
        
        # Normalize labels to AI/Human probabilities
        if label.upper() in ['AI', 'MACHINE', 'GENERATED', 'FAKE', 'LABEL_1', '1']:
            ai_prob = score * 100
            human_prob = (1 - score) * 100
        elif label.upper() in ['HUMAN', 'REAL', 'AUTHENTIC', 'LABEL_0', '0']:
            ai_prob = (1 - score) * 100
            human_prob = score * 100
        else:
            # Default handling
            ai_prob = score * 100
            human_prob = (1 - score) * 100
        
        return ai_prob, human_prob, label, score
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return 50.0, 50.0, "ERROR", 0.5

# Main app
def main():
    # Update page title and subtitle
    st.title("🤖 Advanced AI Text Detector")
    st.subheader("Powered by Two Reliable AI Detection Models")
    
    # Initialize models
    with st.spinner("🔄 Loading AI detection models... This may take a few minutes on first run."):
        pipelines, loading_status = initialize_pipelines()
    
    # Show loading status
    st.subheader("📊 Model Loading Status")
    status_col1, status_col2 = st.columns(2)
    
    with status_col1:
        st.write("**🤖 ChatGPT Detector**")
        st.write(loading_status.get('chatgpt', '❓ Unknown'))
    
    with status_col2:
        st.write("**🔍 OpenAI Detector**")
        st.write(loading_status.get('openai', '❓ Unknown'))
    
    if len(pipelines) == 0:
        st.error("❌ No models could be loaded. Please check your internet connection and try again.")
        st.stop()
    
    st.success(f"✅ {len(pipelines)} model(s) loaded successfully!")
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    
    # Model selection
    available_models = {
        "chatgpt": "🤖 ChatGPT Detector RoBERTa", 
        "openai": "🔍 OpenAI Community RoBERTa"
    }
    
    working_models = {k: v for k, v in available_models.items() if k in pipelines}
    
    if len(working_models) > 1:
        model_choice = st.sidebar.selectbox(
            "Choose AI Detection Model:",
            options=list(working_models.keys()),
            format_func=lambda x: working_models[x]
        )
    else:
        model_choice = list(working_models.keys())[0]
        st.sidebar.write(f"Using: {working_models[model_choice]}")
    
    # Model information
    st.sidebar.subheader("📋 Model Information")
    model_descriptions = {
        "chatgpt": "RoBERTa-based model fine-tuned for detecting ChatGPT content.",
        "openai": "Official OpenAI community model for AI text detection."
    }
    
    if model_choice in model_descriptions:
        st.sidebar.info(model_descriptions[model_choice])
    
    # Main interface
    st.header("📝 Text Analysis")
    
    # Text input
    user_input = st.text_area(
        "Enter text to analyze:",
        height=150,
        placeholder="Paste your text here to check if it's AI-generated or human-written..."
    )
    
    # Analysis options
    col1, col2 = st.columns([1, 1])
    
    with col1:
        analyze_single = st.button("🔍 Analyze with Selected Model", type="primary")
    
    with col2:
        analyze_all = st.button("🚀 Analyze with All Models")
    
    # Analysis results
    if user_input.strip() and (analyze_single or analyze_all):
        text_to_analyze = user_input.strip()
        
        if len(text_to_analyze) < 10:
            st.warning("⚠️ Please enter more text for better analysis (at least 10 characters)")
        else:
            if analyze_single:
                # Single model analysis
                if model_choice in pipelines:
                    st.subheader(f"📊 Results from {working_models[model_choice]}")
                    
                    with st.spinner("Analyzing..."):
                        ai_prob, human_prob, label, score = predict_with_pipeline(
                            pipelines[model_choice], text_to_analyze
                        )
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("🤖 AI-Generated", f"{ai_prob:.1f}%")
                    with col2:
                        st.metric("👤 Human-Written", f"{human_prob:.1f}%")
                    
                    # Final prediction
                    if ai_prob > human_prob:
                        st.error(f"**🤖 Prediction: AI-Generated** (Confidence: {ai_prob:.1f}%)")
                    else:
                        st.success(f"**👤 Prediction: Human-Written** (Confidence: {human_prob:.1f}%)")
                    
                    # Progress bar
                    st.subheader("📈 Confidence Visualization")
                    st.progress(float(ai_prob) / 100)
                    st.caption(f"AI Likelihood: {ai_prob:.1f}% | Human Likelihood: {human_prob:.1f}%")
                    
                    # Raw output
                    with st.expander("🔧 Raw Model Output"):
                        st.write(f"Original Label: {label}")
                        st.write(f"Original Score: {score:.4f}")
            
            elif analyze_all:
                # All models analysis
                st.subheader("🚀 Comparison Across All Models")
                
                results = {}
                
                for model_key, model_name in working_models.items():
                    if model_key in pipelines:
                        with st.spinner(f"Analyzing with {model_name}..."):
                            ai_prob, human_prob, label, score = predict_with_pipeline(
                                pipelines[model_key], text_to_analyze
                            )
                            results[model_key] = {
                                'name': model_name,
                                'ai_prob': ai_prob,
                                'human_prob': human_prob,
                                'prediction': 'AI-Generated' if ai_prob > human_prob else 'Human-Written'
                            }
                
                # Display comparison
                if results:
                    # Create comparison table
                    comparison_data = []
                    for model_key, result in results.items():
                        comparison_data.append({
                            'Model': result['name'],
                            'Prediction': result['prediction'],
                            'AI Probability': f"{result['ai_prob']:.1f}%",
                            'Human Probability': f"{result['human_prob']:.1f}%"
                        })
                    
                    df_comparison = pd.DataFrame(comparison_data)
                    st.dataframe(df_comparison, use_container_width=True)
                    
                    # Consensus
                    ai_predictions = sum(1 for r in results.values() if r['ai_prob'] > r['human_prob'])
                    total_models = len(results)
                    
                    st.subheader("🎯 Consensus")
                    if ai_predictions > total_models / 2:
                        st.error(f"**🤖 Consensus: AI-Generated** ({ai_predictions}/{total_models} models agree)")
                    elif ai_predictions < total_models / 2:
                        st.success(f"**👤 Consensus: Human-Written** ({total_models - ai_predictions}/{total_models} models agree)")
                    else:
                        st.warning(f"**🤔 No Clear Consensus** (Split decision: {ai_predictions}/{total_models})")
    
    # Footer
    st.markdown("---")
    st.markdown("### 📚 About the Models")
    
    info_col1, info_col2 = st.columns(2)
    
    with info_col1:
        st.markdown("""
        **🤖 ChatGPT Detector RoBERTa**
        - ChatGPT-specific detection
        - RoBERTa architecture
        - Fine-tuned for GPT content
        """)
    
    with info_col2:
        st.markdown("""
        **🔍 OpenAI Community RoBERTa**
        - Official OpenAI community model
        - Broad AI text detection
        - Robust across different AI sources
        """)

if __name__ == "__main__":
    main()
