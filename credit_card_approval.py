# app.py
"""
Main Streamlit Application - Credit Card Approval Predictor
"""
import logging
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

# Local imports
from config import FEATURE_GROUPS, CATEGORICAL_MAPPINGS, DATA_SOURCES
from data_processor import load_data, create_pipeline, ChangeToNumTarget
from model_loader import make_prediction
from ui_components import create_input_form, display_prediction

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Credit Guru",
    page_icon="💳",
    layout="centered",
    initial_sidebar_state="collapsed"
)


def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'prediction_made' not in st.session_state:
        st.session_state.prediction_made = False
    if 'last_prediction' not in st.session_state:
        st.session_state.last_prediction = None


@st.cache_data
def prepare_data():
    """Load and prepare training data"""
    train_df, test_df = load_data()
    full_df = pd.concat([train_df, test_df], axis=0).sample(frac=1).reset_index(drop=True)
    
    # Split data
    from sklearn.model_selection import train_test_split
    train_data, test_data = train_test_split(full_df, test_size=0.2, random_state=42)
    
    return train_data, test_data


def preprocess_prediction_input(inputs: dict, train_columns: pd.Index) -> pd.DataFrame:
    """
    Transform user inputs into the format expected by the pipeline
    """
    # Map categorical values
    mappings = {
        'Marital status': CATEGORICAL_MAPPINGS['Marital status'],
        'Dwelling': CATEGORICAL_MAPPINGS['Dwelling'],
        'Employment status': CATEGORICAL_MAPPINGS['Employment status'],
        'Education level': CATEGORICAL_MAPPINGS['Education level'],
    }
    
    # Build profile row
    profile_data = {
        'ID': 0,
        'Gender': inputs['gender'],
        'Has a car': 'Y' if inputs['car_ownership'] else 'N',
        'Has a property': 'Y' if inputs['property_ownership'] else 'N',
        'Children count': 0,  # Will be dropped
        'Income': inputs['income'],
        'Employment status': mappings['Employment status'][inputs['employment_status']],
        'Education level': mappings['Education level'][inputs['education']],
        'Marital status': mappings['Marital status'][inputs['marital_status']],
        'Dwelling': mappings['Dwelling'][inputs['dwelling']],
        'Age': inputs['age'],
        'Employment length': inputs['employment_length'],
        'Has a mobile phone': 1,  # Will be dropped
        'Has a work phone': inputs['work_phone'],
        'Has a phone': inputs['phone'],
        'Has an email': inputs['email'],
        'Job title': 'to_be_dropped',  # Will be dropped
        'Family member count': inputs['family_count'],
        'Account age': 0.00,  # Will be dropped
        'Is high risk': 0,  # Placeholder
    }
    
    return pd.DataFrame([profile_data], columns=train_columns)


def main():
    """Main application entry point"""
    initialize_session_state()
    
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h1>💳 Credit Guru</h1>
        <p style="color: gray;">Get instant insights on your credit card approval chances</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Initializing..."):
        train_data, _ = prepare_data()
    
    # Create tabs
    input_tab, info_tab = st.tabs(["Application Form", "How It Works"])
    
    with input_tab:
        # User inputs
        inputs = create_input_form()
        
        # Predict button
        if st.button("🚀 Check For Approval", type="primary", use_container_width=True):
            with st.spinner("Analyzing your profile..."):
                try:
                    # Validate class distribution
                    class_dist = train_data[FEATURE_GROUPS['target']].value_counts()
                    if len(class_dist) < 2:
                        st.error("Model is not ready. Please check training data.")
                        logger.error("Insufficient classes in training data")
                        return
                    
                    # Prepare input
                    profile_df = preprocess_prediction_input(inputs, train_data.columns)
                    
                    # Run pipeline
                    pipeline = create_pipeline()
                    train_with_profile = pd.concat([train_data, profile_df], ignore_index=True)
                    processed_data = pipeline.fit_transform(train_with_profile)
                    
                    # Extract profile
                    profile_processed = processed_data[processed_data['ID'] == 0].drop(
                        columns=['ID', FEATURE_GROUPS['target']])
                    
                    # Make prediction
                    prediction = make_prediction(profile_processed)
                    st.session_state.prediction_made = True
                    st.session_state.last_prediction = prediction
                    
                    display_prediction(prediction)
                    
                except Exception as e:
                    logger.error(f"Prediction error: {e}", exc_info=True)
                    st.error("An error occurred. Please try again later.")
    
    with info_tab:
        st.markdown("""
        ### How It Works
        
        This app uses machine learning to predict credit card approval based on:
        
        - **Personal Information**: Age, gender, marital status
        - **Financial Profile**: Income, employment, education
        - **Assets**: Car and property ownership
        - **Contact Info**: Phone, email availability
        
        ### Data Privacy
        
        - No personal data is stored
        - All processing happens in real-time
        - Model runs on secure AWS infrastructure
        
        ### Model Accuracy
        
        The model achieves ~95% accuracy on historical data.
        However, this is not financial advice.
        """)
        
        # Show class distribution
        st.write("#### Training Data Distribution")
        class_dist = train_data[FEATURE_GROUPS['target']].value_counts()
        st.bar_chart(class_dist.rename({0: 'Approved', 1: 'Rejected'}))
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; font-size: 0.8em; color: gray;">
        Built with ❤️ using Python, Streamlit & AWS | v1.0.0
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
