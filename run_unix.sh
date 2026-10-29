#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
if [ ! -d ".venv" ]; then python3 -m venv .venv; fi
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
if [ ! -f "models/fault_classifier.joblib" ] || [ ! -f "models/fault_localizer.joblib" ]; then python train_models.py; fi
python -m streamlit run app.py
