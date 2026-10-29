from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from joblib import dump

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
DATA.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)
SEED = 42
FAULTS = ["Normal", "Line-to-Ground", "Line-to-Line", "Three-Phase", "Overload"]
LINES = ["N1-N2","N2-N3","N1-N4","N2-N5","N3-N6","N4-N5","N5-N6","N4-N7","N5-N8","N6-PV1","N7-N8","N8-PV1"]

FEATURES = [
    "voltage_sag_pu", "current_multiple", "phase_imbalance", "frequency_deviation_hz",
    "line_loading_pct", "neighbor_anomaly", "solar_output_mw", "battery_soc_pct",
    "sensor_noise", "thermal_margin_pct", "fault_duration_s"
]

def generate_dataset(n=10000, seed=SEED):
    rng = np.random.default_rng(seed)
    labels = rng.choice(FAULTS, size=n, p=[.24,.24,.18,.16,.18])
    line_ids = rng.integers(0, len(LINES), size=n)
    solar = rng.uniform(0, 3, n)
    soc = rng.uniform(15, 100, n)
    noise = rng.uniform(0.0, .12, n)
    voltage = np.empty(n); current = np.empty(n); imbalance = np.empty(n)
    freq = np.empty(n); loading = np.empty(n); neighbor = np.empty(n)
    thermal = np.empty(n); duration = np.empty(n)
    for i, lab in enumerate(labels):
        if lab == "Normal":
            voltage[i] = rng.normal(.99,.025); current[i] = rng.normal(1.0,.12)
            imbalance[i] = abs(rng.normal(.025,.018)); freq[i] = rng.normal(0,.025)
            loading[i] = rng.normal(62,12); neighbor[i] = abs(rng.normal(.08,.04))
            thermal[i] = rng.normal(38,12); duration[i] = rng.uniform(.1,3)
        elif lab == "Line-to-Ground":
            voltage[i] = rng.normal(.72,.09); current[i] = rng.normal(2.25,.45)
            imbalance[i] = abs(rng.normal(.42,.13)); freq[i] = rng.normal(-.06,.06)
            loading[i] = rng.normal(108,20); neighbor[i] = abs(rng.normal(.55,.18))
            thermal[i] = rng.normal(4,12); duration[i] = rng.uniform(.2,12)
        elif lab == "Line-to-Line":
            voltage[i] = rng.normal(.65,.10); current[i] = rng.normal(2.7,.5)
            imbalance[i] = abs(rng.normal(.62,.12)); freq[i] = rng.normal(-.08,.07)
            loading[i] = rng.normal(118,22); neighbor[i] = abs(rng.normal(.7,.16))
            thermal[i] = rng.normal(-3,12); duration[i] = rng.uniform(.2,12)
        elif lab == "Three-Phase":
            voltage[i] = rng.normal(.48,.10); current[i] = rng.normal(3.5,.65)
            imbalance[i] = abs(rng.normal(.13,.08)); freq[i] = rng.normal(-.15,.08)
            loading[i] = rng.normal(145,25); neighbor[i] = abs(rng.normal(.9,.1))
            thermal[i] = rng.normal(-18,14); duration[i] = rng.uniform(.1,8)
        else: # overload
            voltage[i] = rng.normal(.88,.06); current[i] = rng.normal(1.75,.28)
            imbalance[i] = abs(rng.normal(.07,.04)); freq[i] = rng.normal(-.025,.04)
            loading[i] = rng.normal(128,16); neighbor[i] = abs(rng.normal(.3,.12))
            thermal[i] = rng.normal(-9,10); duration[i] = rng.uniform(1,20)
    # Add small measurement noise consistently to all samples.
    voltage += rng.normal(0,.025,n) + noise*.08
    current += rng.normal(0,.06,n)
    imbalance = np.clip(imbalance + rng.normal(0,.02,n),0,1)
    freq += rng.normal(0,.012,n)
    loading = np.clip(loading + rng.normal(0,5,n),0,200)
    neighbor = np.clip(neighbor + rng.normal(0,.025,n),0,1.5)
    thermal = np.clip(thermal + rng.normal(0,4,n),-50,100)
    duration = np.clip(duration,0,30)
    df = pd.DataFrame({
        "voltage_sag_pu":np.clip(voltage,.1,1.15), "current_multiple":np.clip(current,.1,5),
        "phase_imbalance":imbalance, "frequency_deviation_hz":freq,
        "line_loading_pct":loading, "neighbor_anomaly":neighbor, "solar_output_mw":solar,
        "battery_soc_pct":soc, "sensor_noise":noise, "thermal_margin_pct":thermal,
        "fault_duration_s":duration, "fault_type":labels,
        "faulted_line":[LINES[k] for k in line_ids]
    })
    return df

def metrics_for(model, X_test, y_test, labels):
    pred = model.predict(X_test)
    return {
        "accuracy": round(float(accuracy_score(y_test,pred)),4),
        "precision_macro": round(float(precision_score(y_test,pred,average="macro",zero_division=0)),4),
        "recall_macro": round(float(recall_score(y_test,pred,average="macro",zero_division=0)),4),
        "f1_macro": round(float(f1_score(y_test,pred,average="macro",zero_division=0)),4),
        "labels": list(labels),
        "confusion_matrix": confusion_matrix(y_test,pred,labels=labels).tolist(),
        "classification_report": classification_report(y_test,pred,labels=labels,output_dict=True,zero_division=0)
    }

def main():
    df = generate_dataset()
    df.to_csv(DATA/"grid_fault_dataset.csv",index=False)
    # fault classifier
    X = df[FEATURES]; y = df["fault_type"]
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=.2,random_state=SEED,stratify=y)
    clf = RandomForestClassifier(n_estimators=140,max_depth=16,min_samples_leaf=2,class_weight="balanced",random_state=SEED,n_jobs=-1)
    clf.fit(Xtr,ytr); dump(clf,MODELS/"fault_classifier.joblib")
    clf_metrics = metrics_for(clf,Xte,yte,FAULTS)
    clf_metrics["training_samples"] = int(len(Xtr)); clf_metrics["test_samples"] = int(len(Xte))
    clf_metrics["features"] = FEATURES
    # localization model: intentionally no line ID as a direct feature; measured location signature is synthetic proxy
    rng = np.random.default_rng(SEED+1)
    loc_df = df[FEATURES].copy()
    # Simulated sensor signature for each of 12 feeder segments, measured with noise.
    for j,line in enumerate(LINES):
        loc_df[f"segment_sensor_{j+1:02d}"] = (df["faulted_line"].eq(line).astype(float) * 0.55 +
                                                 rng.normal(0,.24,len(df)) +
                                                 df["neighbor_anomaly"].to_numpy()*.05)
    loc_features = list(loc_df.columns)
    lxtr,lxte,lytr,lyte = train_test_split(loc_df,df["faulted_line"],test_size=.2,random_state=SEED,stratify=df["faulted_line"])
    loc = RandomForestClassifier(n_estimators=140,max_depth=18,min_samples_leaf=1,class_weight="balanced",random_state=SEED,n_jobs=-1)
    loc.fit(lxtr,lytr); dump(loc,MODELS/"fault_localizer.joblib")
    loc_metrics = metrics_for(loc,lxte,lyte,LINES)
    loc_metrics["training_samples"] = int(len(lxtr)); loc_metrics["test_samples"] = int(len(lxte))
    loc_metrics["features"] = loc_features
    report = {
        "generated_at_note":"Reproducible synthetic dataset; metrics are from a held-out synthetic test split, not real grid measurements.",
        "dataset_rows":int(len(df)), "random_seed":SEED,
        "fault_classifier":clf_metrics, "fault_localizer":loc_metrics,
        "fault_labels":FAULTS, "line_labels":LINES, "fault_features":FEATURES,
        "localization_features":loc_features
    }
    (MODELS/"model_metrics.json").write_text(json.dumps(report,indent=2))
    print(f"Generated {len(df):,} synthetic samples.")
    print(f"Fault classifier: accuracy={clf_metrics['accuracy']:.3f}, macro-F1={clf_metrics['f1_macro']:.3f}")
    print(f"Fault localizer:   accuracy={loc_metrics['accuracy']:.3f}, macro-F1={loc_metrics['f1_macro']:.3f}")
    print("Saved dataset, trained models, and metrics.")
if __name__ == "__main__":
    main()
