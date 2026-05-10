import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import sys
sys.path.append('..')
from utils import clean_text_classification, categorize_activity, train_classification_model, predict_activity

st.set_page_config(page_title="Activity Classification", layout="wide")

st.title("Activity Classification")
st.markdown("Automatically classify job activities into categories using Machine Learning")

st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input Data")
    
    input_method = st.radio("Input Method:", ["Single Activity", "Upload CSV"])
    
    if input_method == "Single Activity":
        activity_text = st.text_area(
            "Enter activity description:",
            placeholder="e.g., Develop Python applications and manage SQL databases...",
            height=150
        )
        
        if st.button("Classify Activity"):
            if activity_text.strip():
                # Use rule-based classification
                category = categorize_activity(activity_text)
                st.session_state.predicted_category = category
                st.session_state.activity_input = activity_text
                st.success("Classification complete!")
            else:
                st.warning("Please enter an activity description")
    
    else:  # Upload CSV
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv', 'xlsx'])
        
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.write(f"Loaded {len(df)} rows")
                st.dataframe(df.head(), use_container_width=True)
                
                text_column = st.selectbox("Select text column:", df.columns)
                
                if st.button("Classify All Activities"):
                    # Prepare data
                    df['clean_text'] = df[text_column].apply(clean_text_classification)
                    df['category'] = df['clean_text'].apply(categorize_activity)
                    
                    # Train model on this data
                    try:
                        model, vectorizer, encoder = train_classification_model(df)
                        st.session_state.model = model
                        st.session_state.vectorizer = vectorizer
                        st.session_state.encoder = encoder
                        st.session_state.df_classified = df
                        st.success("All activities classified!")
                    except Exception as e:
                        st.error(f"Error during classification: {e}")
            
            except Exception as e:
                st.error(f"Error loading file: {e}")

with col2:
    st.subheader("Results")
    
    if input_method == "Single Activity" and "predicted_category" in st.session_state:
        category = st.session_state.predicted_category
        
        st.markdown(f"### Predicted Category")
        st.success(f"**{category}**")
        
        st.info(f"Activity: {st.session_state.activity_input[:200]}...")
    
    elif input_method == "Upload CSV" and "df_classified" in st.session_state:
        df = st.session_state.df_classified
        
        st.write(f"**Classification Results:**")
        st.dataframe(df[['clean_text', 'category']], use_container_width=True)
        
        # Category distribution
        st.markdown("---")
        category_counts = df['category'].value_counts()
        
        fig, ax = plt.subplots(figsize=(8, 5))
        category_counts.plot(kind='bar', ax=ax, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax.set_title('Distribution of Activity Categories')
        ax.set_xlabel('Category')
        ax.set_ylabel('Count')
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        # Metrics
        col_metric1, col_metric2, col_metric3 = st.columns(3)
        with col_metric1:
            st.metric("Total Activities", len(df))
        with col_metric2:
            st.metric("Unique Categories", df['category'].nunique())
        with col_metric3:
            st.metric("Most Common", category_counts.index[0])
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Results (CSV)",
            data=csv,
            file_name="activity_classification_results.csv",
            mime="text/csv"
        )

st.markdown("---")
st.info("""
**Classification Categories:**
- **Technical**: Development, data science, programming, engineering
- **Managerial**: Project management, leadership, planning, strategy
- **Soft Skills**: Communication, teamwork, problem solving, collaboration
- **Other**: Everything else

**How it works:**
1. Enter activity descriptions or upload a CSV
2. System uses rule-based + ML classification
3. Activities are categorized automatically
4. View metrics and download results
""")
