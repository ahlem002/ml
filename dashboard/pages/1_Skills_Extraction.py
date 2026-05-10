import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import sys
sys.path.append('..')
from utils import load_spacy_model, extract_skills, clean_text

st.set_page_config(page_title="Skills Extraction", page_icon="🔍", layout="wide")

st.title("🔍 Skills Extraction")
st.markdown("Extract technical, managerial, and soft skills from job descriptions using NLP")

st.markdown("---")

# Sidebar
with st.sidebar:
    st.subheader("⚙️ Settings")
    input_method = st.radio("Input Method:", ["Text Input", "Upload CSV"])

# Initialize spaCy model
nlp = load_spacy_model()

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📥 Input Data")
    
    if input_method == "Text Input":
        user_text = st.text_area(
            "Enter job description:",
            placeholder="Paste job description here...",
            height=200,
            key="job_desc"
        )
        
        if st.button("🔍 Extract Skills", key="extract_btn"):
            if user_text.strip():
                clean_user_text = clean_text(user_text)
                skills = extract_skills(clean_user_text, nlp)
                
                st.session_state.extracted_skills = skills
                st.session_state.user_text = user_text
                st.success("✅ Skills extracted successfully!")
            else:
                st.warning("Please enter a job description")
    
    else:  # Upload CSV
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv', 'xlsx'])
        
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.write(f"✅ Loaded {len(df)} rows")
                st.dataframe(df.head(), use_container_width=True)
                
                text_column = st.selectbox("Select text column:", df.columns)
                
                if st.button("🔍 Extract Skills from All Rows"):
                    progress_bar = st.progress(0)
                    all_skills = []
                    
                    for idx, row in df.iterrows():
                        text = row[text_column]
                        clean_text_str = clean_text(str(text))
                        skills = extract_skills(clean_text_str, nlp)
                        
                        all_skills.append({
                            'row_id': idx,
                            'text': text,
                            'technical_skills': ', '.join(skills.get('TECHNICAL_SKILL', [])),
                            'managerial_skills': ', '.join(skills.get('MANAGERIAL_SKILL', [])),
                            'soft_skills': ', '.join(skills.get('SOFT_SKILL', []))
                        })
                        
                        progress_bar.progress((idx + 1) / len(df))
                    
                    st.session_state.df_results = pd.DataFrame(all_skills)
                    st.success("✅ Skills extracted from all rows!")
            
            except Exception as e:
                st.error(f"❌ Error loading file: {e}")

with col2:
    st.subheader("📊 Results")
    
    if input_method == "Text Input" and "extracted_skills" in st.session_state:
        skills = st.session_state.extracted_skills
        
        st.info("**Extracted Skills by Category:**")
        
        col_res1, col_res2, col_res3 = st.columns(3)
        
        with col_res1:
            st.markdown("### 💻 Technical Skills")
            tech_skills = skills.get('TECHNICAL_SKILL', [])
            if tech_skills:
                for skill in tech_skills:
                    st.write(f"• {skill}")
            else:
                st.write("No technical skills found")
        
        with col_res2:
            st.markdown("### 👔 Managerial Skills")
            mgmt_skills = skills.get('MANAGERIAL_SKILL', [])
            if mgmt_skills:
                for skill in mgmt_skills:
                    st.write(f"• {skill}")
            else:
                st.write("No managerial skills found")
        
        with col_res3:
            st.markdown("### 🤝 Soft Skills")
            soft_skills = skills.get('SOFT_SKILL', [])
            if soft_skills:
                for skill in soft_skills:
                    st.write(f"• {skill}")
            else:
                st.write("No soft skills found")
        
        # Summary
        st.markdown("---")
        total_skills = sum(len(v) for v in skills.values())
        st.metric("Total Skills Found", total_skills)
    
    elif input_method == "Upload CSV" and "df_results" in st.session_state:
        st.dataframe(st.session_state.df_results, use_container_width=True)
        
        # Download button
        csv = st.session_state.df_results.to_csv(index=False)
        st.download_button(
            label="📥 Download Results (CSV)",
            data=csv,
            file_name="skills_extraction_results.csv",
            mime="text/csv"
        )

st.markdown("---")
st.info("""
**How it works:**
1. Enter a job description or upload a CSV file
2. The system uses spaCy NLP to extract skills
3. Skills are categorized as: Technical, Managerial, or Soft Skills
4. Download results for further analysis
""")
