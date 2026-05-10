import streamlit as st
from PIL import Image
import os

# Page Configuration
st.set_page_config(
    page_title="HR & Skills Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .subheader {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Main Page
st.markdown('<div class="main-header">HR & Skills Analytics Dashboard</div>', unsafe_allow_html=True)

st.markdown("""
---
### Welcome to the Dashboard

This dashboard provides 3 powerful analytics modules for HR and Skills Management:

**Navigate using the sidebar menu to access:**
""")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### Page 1: Skills Extraction
    **Automatic Skills Extraction from Job Descriptions**
    
    - Extract technical, managerial, and soft skills
    - NLP-powered using spaCy
    - Classify skills by category
    - Upload job descriptions (text/CSV)
    """)

with col2:
    st.markdown("""
    ### Page 2: Activity Classification
    **Classify Job Activities Automatically**
    
    - Categorize activities (Technical/Managerial/Soft Skills)
    - Machine learning-powered classification
    - View confusion matrix & accuracy metrics
    - Support for CSV data input
    """)

with col3:
    st.markdown("""
    ### Page 3: Employee Clustering
    **Analyze Employee Segments**
    
    - K-Means & Hierarchical clustering
    - Visualize clusters with PCA
    - HR data analysis & insights
    - Upload HR datasets (CSV)
    """)

st.markdown("---")

st.info("""
**How to Use:**
1. Select a page from the sidebar menu
2. Upload your data or use sample data
3. Review results and visualizations
4. Download analysis results
""")
