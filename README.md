# GridGuard AI — Full Runnable Web Application

GridGuard AI is an educational/research demonstration of AI-assisted fault detection, fault localization, recovery candidate generation, and constraint-aware recovery recommendations for a **simulated** distribution grid.

## What is included
- Responsive Streamlit web application with animated workflow, interactive network map, fault injection, recovery candidates, solar/battery controls, before/after view, metrics, and event log.
- A reproducible synthetic dataset generator (`data/grid_fault_dataset.csv`).
- Two trained scikit-learn Random Forest models:
  1. fault classification (`models/fault_classifier.joblib`)
  2. faulted-line localization (`models/fault_localizer.joblib`)
- Training metrics, confusion matrices, and classification reports (`models/model_metrics.json`).
- Automated smoke tests.

## Quick start

### Windows
1. Install **Python 3.11–3.13** from https://www.python.org/downloads/ and enable **Add Python to PATH**.
2. Extract this ZIP.
3. Double-click `run_windows.bat`.
4. Wait for dependencies/model training to finish on first run.
5. Open the local URL printed in the terminal (usually `http://localhost:8501`).

### macOS / Linux
From the extracted folder:
```bash
chmod +x run_unix.sh
./run_unix.sh
```
Or run manually:
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

Internet access is needed for the first dependency installation. After installation, the app and models run locally.

## Using the app
1. Select a faulted line and fault type in the sidebar.
2. Adjust solar generation and battery support.
3. Click **Inject Fault** or **Run Full Self-Healing Demo**.
4. Review detection/localization outputs and the generated recovery plans.
5. Inspect the validation reasons and before/after grid state.

## Models and evaluation
`train_models.py` generates a deterministic synthetic dataset, trains both models using a stratified train/test split, and exports accuracy, precision, recall, F1, confusion matrices, and classification reports. The app reads those metrics from `models/model_metrics.json`.

**Important:** Metrics are measured on a held-out split of synthetic data only. They are not evidence of performance on a real utility grid. The localization model predicts a simulated line label, not a physical location measured in the field.

## Safety and limitations
- This is a **simulation-only educational prototype**. It is not connected to SCADA, real sensors, breakers, or utility systems.
- The network visual and constraint checks are illustrative. They are not a validated AC power-flow or protection study.
- Do not use the suggested switching actions on real electrical equipment.
- The project uses Random Forest models; it does not claim to contain a trained GNN or reinforcement-learning agent. Recovery planning uses explicit candidate generation and simplified constraint checks.
- To progress toward research-grade validation, integrate a validated simulator such as pandapower, use a standard test feeder, document baselines, and have power-systems specialists review the assumptions.

## Reproducibility
Run `python train_models.py` at any time to regenerate the dataset and models from the fixed random seed.
