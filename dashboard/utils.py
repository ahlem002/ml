import re
import spacy
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import streamlit as st

# ============== SKILLS EXTRACTION UTILITIES ==============

@st.cache_resource
def load_spacy_model():
    """Load spaCy model for NER.

    On Streamlit Cloud the model must be installed via requirements.txt
    (using the wheel URL from the spacy-models GitHub releases) because
    the runtime environment is sandboxed and `spacy download` may fail.
    """
    try:
        return spacy.load('en_core_web_sm')
    except OSError:
        # Fallback: attempt a runtime install via pip (works locally,
        # may fail on Streamlit Cloud).
        import subprocess
        import sys as _sys

        st.warning("spaCy model not found. Attempting to install at runtime...")
        wheel_url = (
            "https://github.com/explosion/spacy-models/releases/download/"
            "en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
        )
        try:
            subprocess.check_call(
                [_sys.executable, "-m", "pip", "install", wheel_url]
            )
            return spacy.load('en_core_web_sm')
        except Exception as exc:
            st.error(
                "Could not load the spaCy model `en_core_web_sm`. "
                "Add the model wheel URL to `requirements.txt`:\n\n"
                "`https://github.com/explosion/spacy-models/releases/"
                "download/en_core_web_sm-3.8.0/"
                "en_core_web_sm-3.8.0-py3-none-any.whl`\n\n"
                f"Underlying error: {exc}"
            )
            st.stop()

# Skills are listed as plain strings (single or multi-word). Patterns
# are built automatically from these lists, which makes the dictionary
# easy to extend.

TECHNICAL_SKILLS = [
    # Programming languages
    "python", "sql", "java", "javascript", "typescript", "c++", "c#", "go",
    "ruby", "scala", "r", "matlab", "kotlin", "swift", "php", "rust",
    # Web / frameworks
    "react", "angular", "vue", "node", "node.js", "django", "flask",
    "fastapi", "spring", "express", "html", "css",
    # Cloud / infra / devops
    "docker", "kubernetes", "aws", "azure", "gcp", "google cloud",
    "terraform", "ansible", "jenkins", "ci/cd", "devops", "linux",
    # Data / ML / AI
    "machine learning", "deep learning", "data science", "data analysis",
    "data engineering", "artificial intelligence", "nlp",
    "natural language processing", "computer vision", "statistics",
    "regression", "classification", "clustering", "tensorflow", "pytorch",
    "keras", "pandas", "numpy", "scikit-learn", "spark", "hadoop", "kafka",
    "airflow", "etl",
    # BI / databases
    "tableau", "power bi", "excel", "mysql", "postgresql", "mongodb",
    "nosql", "mongodb", "snowflake", "databricks",
    # Engineering / tooling
    "git", "github", "gitlab", "rest api", "graphql", "microservices",
    # Domain-specific technical
    "haccp", "fda", "gmp", "iso", "six sigma", "lean", "kaizen",
    "manufacturing", "production", "operations", "engineering",
    "quality control", "quality assurance", "regulatory compliance",
    "audits", "auditing", "process improvement",
]

MANAGERIAL_SKILLS = [
    "leadership", "management", "manager", "managerial",
    "project management", "team management", "people management",
    "operations management", "program management", "product management",
    "stakeholder management", "vendor management", "change management",
    "risk management", "performance management", "talent management",
    "supply chain management",
    "agile", "scrum", "kanban", "waterfall",
    "budgeting", "budget", "forecasting", "financial planning",
    "strategic planning", "strategy", "business strategy",
    "planning", "execution", "decision making", "decision-making",
    "mentoring", "coaching", "supervision", "supervising",
    "delegation", "prioritization", "oversight", "governance",
    "compliance", "accountability", "recruiting", "hiring",
    "negotiation", "contract negotiation", "stakeholder engagement",
    "process management", "kpi", "okr", "roadmap", "roadmapping",
    "p&l", "business development",
]

SOFT_SKILLS = [
    "communication", "written communication", "verbal communication",
    "interpersonal", "interpersonal skills", "presentation",
    "public speaking", "active listening", "listening",
    "teamwork", "team work", "team player", "collaboration",
    "cooperation", "partnership",
    "problem solving", "problem-solving", "analytical thinking",
    "analytical skills", "critical thinking", "logical thinking",
    "adaptability", "flexibility", "agility",
    "creativity", "creative thinking", "innovation", "innovative",
    "time management", "organization", "organizational skills",
    "organized", "self-organized",
    "attention to detail", "detail oriented", "detail-oriented",
    "emotional intelligence", "empathy",
    "conflict resolution", "conflict management",
    "multitasking", "multi-tasking",
    "self-motivated", "motivation", "motivated", "proactive",
    "proactivity", "initiative",
    "work ethic", "reliability", "reliable",
    "customer service", "customer focus", "client management",
    "relationship building", "relationship management", "networking",
    "cultural awareness", "diversity", "inclusion",
]


def _build_skill_patterns():
    """Build spaCy EntityRuler patterns from the skill lists above."""
    categories = [
        ("TECHNICAL_SKILL", TECHNICAL_SKILLS),
        ("MANAGERIAL_SKILL", MANAGERIAL_SKILLS),
        ("SOFT_SKILL", SOFT_SKILLS),
    ]
    patterns = []
    seen = set()
    for label, skills in categories:
        for skill in skills:
            key = (label, skill.lower())
            if key in seen:
                continue
            seen.add(key)
            tokens = skill.lower().split()
            patterns.append({
                "label": label,
                "pattern": [{"LOWER": tok} for tok in tokens],
            })
    return patterns


SKILL_PATTERNS = _build_skill_patterns()


def extract_skills(text, nlp):
    """Extract skills from text using spaCy EntityRuler.

    Returns a dict keyed by skill category (TECHNICAL_SKILL,
    MANAGERIAL_SKILL, SOFT_SKILL) with deduplicated skill names.
    """
    if not nlp.has_pipe("entity_ruler"):
        ruler = nlp.add_pipe("entity_ruler", before="ner")
        ruler.add_patterns(SKILL_PATTERNS)

    doc = nlp(text.lower())
    skills = {}
    for ent in doc.ents:
        if ent.label_ not in {"TECHNICAL_SKILL", "MANAGERIAL_SKILL", "SOFT_SKILL"}:
            continue
        bucket = skills.setdefault(ent.label_, [])
        if ent.text not in bucket:
            bucket.append(ent.text)

    return skills

def clean_text(text):
    """Clean text for processing"""
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ============== CLASSIFICATION UTILITIES ==============

def clean_text_classification(text):
    """Clean text for classification"""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def categorize_activity(text):
    """Token-based scoring categorization.

    Count whole-word matches per category and choose the highest-scoring category.
    Tie-breaker order: Managerial > Technical > Soft Skills > Other.
    This avoids accidental substring matches and gives more robust results.
    """
    tech = ['python', 'java', 'sql', 'data', 'developer', 'engineering', 'machine learning', 'code', 'programming']
    mg = ['manager', 'lead', 'project', 'scrum', 'director', 'strategy', 'planning', 'leadership']
    soft = ['communication', 'teamwork', 'leadership', 'adaptability', 'problem solving', 'collaboration']

    # Normalize and tokenize (simple whitespace/token split)
    text_lower = str(text).lower()
    # Replace punctuation with spaces, keep alphanumerics and spaces
    text_clean = re.sub(r'[^a-z0-9\s]', ' ', text_lower)
    tokens = [t for t in text_clean.split() if t]

    def count_matches(keywords):
        cnt = 0
        for kw in keywords:
            # match multi-word keywords as phrases
            if ' ' in kw:
                if kw in text_clean:
                    cnt += 1
            else:
                cnt += tokens.count(kw)
        return cnt

    tech_score = count_matches(tech)
    mg_score = count_matches(mg)
    soft_score = count_matches(soft)

    # Choose highest score with tie-breaker preference
    scores = {
        'Managerial': mg_score,
        'Technical': tech_score,
        'Soft Skills': soft_score,
        'Other': 0
    }

    # Select category with highest score; tie-breaker by defined order
    best = max(scores.items(), key=lambda kv: (kv[1], {'Managerial':3,'Technical':2,'Soft Skills':1,'Other':0}[kv[0]]))[0]
    if scores[best] == 0:
        return 'Other'
    return best

@st.cache_resource
def train_classification_model(df, text_col='clean_text', label_col='category'):
    """Train TF-IDF + SVM model"""
    vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
    X = vectorizer.fit_transform(df[text_col])
    
    encoder = LabelEncoder()
    y = encoder.fit_transform(df[label_col])
    
    model = SVC(kernel='linear', class_weight='balanced')
    model.fit(X, y)
    
    return model, vectorizer, encoder

def predict_activity(text, model, vectorizer, encoder):
    """Predict activity category"""
    text_clean = clean_text_classification(text)
    X = vectorizer.transform([text_clean])
    pred = model.predict(X)
    return encoder.inverse_transform(pred)[0]

# ============== CLUSTERING UTILITIES ==============

def prepare_hr_data(df, selected_features):
    """Prepare HR data for clustering"""
    cols_to_drop = ["EmployeeCount", "StandardHours", "Over18", "EmployeeNumber", "Attrition"]
    df_clean = df.drop(columns=cols_to_drop, errors="ignore")
    
    existing_features = [col for col in selected_features if col in df_clean.columns]
    df_selected = df_clean[existing_features].copy()
    
    categorical_cols = [col for col in ["Department", "JobRole"] if col in df_selected.columns]
    df_encoded = pd.get_dummies(df_selected, columns=categorical_cols, drop_first=True)
    df_encoded = df_encoded.astype(int)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_encoded)
    
    return X_scaled, df_selected, df_encoded

def perform_clustering(X_scaled, n_clusters=3):
    """Perform K-Means clustering"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # PCA for visualization
    n_components = min(2, X_scaled.shape[1])
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    return clusters, kmeans, X_pca, pca

def get_silhouette_scores(X_scaled, max_clusters=10):
    """Calculate silhouette scores for different cluster numbers"""
    from sklearn.metrics import silhouette_score
    scores = []
    for n_clusters in range(2, min(max_clusters + 1, len(X_scaled))):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, cluster_labels)
        scores.append((n_clusters, score))
    return scores
