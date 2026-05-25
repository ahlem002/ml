# ML Dashboard and Notebooks

Small project containing data analysis notebooks and a Streamlit dashboard for HR data.

## Project Overview
- Dashboard: interactive web app to extract skills, classify activities, and cluster employees.
- Notebooks: exploratory analysis and model work.

## Repository Structure
- dashboard/: Streamlit app and pages
  - app.py - main Streamlit app
  - requirements.txt - dependencies for the dashboard
  - pages/ - multipage components
- *.ipynb - Jupyter notebooks for experiments and analyses
- *.csv - processed datasets used by notebooks

## Setup (Windows)
1. Create and activate a virtual environment:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.venv\Scripts\Activate.ps1
```

2. Install dependencies for the dashboard:

```powershell
pip install -r dashboard\requirements.txt
```

## Run the dashboard

```powershell
streamlit run dashboard\app.py
```



