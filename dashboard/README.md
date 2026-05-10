# 📊 HR & Skills Analytics Dashboard

A comprehensive Streamlit dashboard with 3 integrated pages for HR and Skills analysis.

## 🎯 Dashboard Pages

### 1️⃣ **Page 1: Skills Extraction** (`1_Skills_Extraction.py`)
Extract technical, managerial, and soft skills from job descriptions using NLP.
- **Features**: 
  - Text input or CSV upload
  - NLP-powered skill extraction using spaCy
  - Categorizes skills automatically
  - Download results as CSV

### 2️⃣ **Page 2: Activity Classification** (`2_Activity_Classification.py`)
Automatically classify job activities into categories.
- **Features**:
  - Single activity prediction
  - Batch CSV processing
  - Rule-based + ML classification
  - Confusion matrix & accuracy metrics
  - Download classified results

### 3️⃣ **Page 3: Employee Clustering** (`3_Employee_Clustering.py`)
Analyze employee segments using K-Means clustering.
- **Features**:
  - HR data clustering
  - PCA 2D visualization
  - Silhouette analysis
  - Cluster statistics
  - Download clustered data

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip

### Installation

1. **Navigate to the dashboard folder:**
```bash
cd dashboard
```

2. **Create a virtual environment (recommended):**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Download spaCy model (for Skills Extraction):**
```bash
python -m spacy download en_core_web_sm
```

### Running the Dashboard

```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`

---

## 📁 Project Structure

```
dashboard/
├── app.py                          # Main entry point (Home page)
├── utils.py                        # Shared utilities & functions
├── requirements.txt                # Python dependencies
├── .streamlit/
│   └── config.toml                # Streamlit configuration
└── pages/
    ├── 1_Skills_Extraction.py      # Page 1: Skills Extraction
    ├── 2_Activity_Classification.py # Page 2: Activity Classification
    └── 3_Employee_Clustering.py    # Page 3: Employee Clustering
```

---

## 📊 Usage Examples

### Page 1: Skills Extraction
**Scenario**: Extract skills from a job posting

```
Input: "We are looking for a Python developer with SQL knowledge, 
excellent communication skills, and project management experience."

Output:
- Technical Skills: python, sql
- Managerial Skills: project management
- Soft Skills: communication
```

### Page 2: Activity Classification
**Scenario**: Classify employee activities

```
Input: "Develop backend APIs using Python and manage SQL databases"

Output: Technical (with 95% confidence)
```

### Page 3: Employee Clustering
**Scenario**: Segment HR data into employee groups

```
Input: HR dataset with 1000 employees
Output: 3 clusters with average silhouette score of 0.65
```

---

## 📝 Input Data Formats

### Page 1 & 2: CSV Format
```csv
description
"python developer with 5 years experience"
"project manager leading 10-person team"
"data scientist specializing in ml"
```

### Page 3: HR CSV Format
```csv
Age,Department,JobRole,MonthlyIncome,Experience
28,Sales,Manager,5000,5
35,IT,Developer,6000,10
42,HR,Analyst,4500,15
```

---

## 🔧 Configuration

Edit `.streamlit/config.toml` to customize:
- Theme colors
- Page layout
- Font settings

Current theme: Coral Red (#FF6B6B)

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| streamlit | 1.28.1 | Web framework |
| pandas | 2.0.3 | Data manipulation |
| scikit-learn | 1.3.0 | ML models |
| spacy | 3.6.0 | NLP processing |
| matplotlib | 3.7.2 | Visualizations |
| seaborn | 0.12.2 | Statistical plots |

---

## ✅ Features Checklist

- [x] Multi-page Streamlit app
- [x] Skills extraction with NLP
- [x] Activity classification with ML
- [x] Employee clustering with visualization
- [x] CSV/Excel file upload support
- [x] Download results as CSV
- [x] Silhouette analysis
- [x] PCA visualization
- [x] Responsive design

---

## 🐛 Troubleshooting

### spaCy model not found
```bash
python -m spacy download en_core_web_sm
```

### Import errors
Reinstall dependencies:
```bash
pip install --upgrade -r requirements.txt
```

### CSV parsing errors
- Ensure CSV is UTF-8 encoded
- Check that required columns exist
- Try different delimiters

---

## 📄 Sample Datasets

You can use the provided datasets:
- `Cleaned_HR_Data_Analysis.csv` → Page 3
- `JobsDatasetProcessed (2).csv` → Page 2
- Custom text data → Page 1

---

## 🎨 Customization

### Add custom skills patterns
Edit `utils.py` → `SKILL_PATTERNS` list

### Change clustering algorithm
Edit `utils.py` → `perform_clustering()` function

### Modify theme colors
Edit `.streamlit/config.toml` → `[theme]` section

---

## 📞 Support

For issues or questions:
1. Check the [Streamlit documentation](https://docs.streamlit.io)
2. Review the code comments in each page
3. Check the sidebar "ℹ️ Info" sections on each page

---

## 📄 License

This project is provided as-is for educational and business purposes.

---

**Created with ❤️ using Streamlit, scikit-learn, and spaCy**
