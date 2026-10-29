from pathlib import Path
import json, subprocess, sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from joblib import load

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
DATA = ROOT / "data"
st.set_page_config(page_title="GridGuard AI", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

# ---------- Visual design ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { --gg-ink:#0d1729; --gg-blue:#102744; --gg-cyan:#0eafc4; --gg-green:#249b52; --gg-muted:#64748b; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background:linear-gradient(180deg,#f7fafc 0%,#eef4f8 100%); color:#101828; }
h1,h2,h3 { font-family:'Space Grotesk',sans-serif !important; letter-spacing:-.035em; }
.block-container { max-width:1440px; padding-top:1.5rem; padding-bottom:4rem; }
[data-testid="stSidebar"] { background:#0d1b2e; }
[data-testid="stSidebar"] * { color:#eef6ff; }
[data-testid="stSidebar"] [data-baseweb="select"] * { color:#111827; }

/* Fix dropdown visibility in the sidebar */
[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] {
    background-color: #ffffff !important;
    border-color: #cbd5e1 !important;
}

[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] * {
    color: #111827 !important;
    -webkit-text-fill-color: #111827 !important;
}

[data-testid="stSidebar"] [data-testid="stSelectbox"] input {
    color: #111827 !important;
    -webkit-text-fill-color: #111827 !important;
}

[data-baseweb="popover"],
[data-baseweb="menu"],
[role="listbox"] {
    background-color: #ffffff !important;
}

[role="option"],
[role="option"] * {
    color: #111827 !important;
    background-color: #ffffff;
}

[role="option"]:hover {
    background-color: #e6f7fb !important;
}

.gg-hero { border-radius:24px; padding:28px 32px; color:white; background:radial-gradient(circle at 85% 15%,rgba(14,175,196,.36),transparent 32%),linear-gradient(125deg,#0b1728,#173c61 65%,#0d5666); box-shadow:0 16px 40px rgba(13,39,68,.16); position:relative; overflow:hidden; animation:rise .65s ease-out; }
.gg-brand { font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:clamp(28px,4vw,42px); letter-spacing:-1.5px; }
.gg-sub { color:#c4d4e5; font-size:15px; margin-top:3px; }
.gg-pill { display:inline-flex; align-items:center; gap:8px; border:1px solid rgba(94,234,142,.35); border-radius:999px; padding:7px 13px; background:rgba(22,163,74,.14); color:#a7f3c0; font-weight:700; font-size:12px; letter-spacing:.06em; }
.gg-dot { width:8px; height:8px; border-radius:50%; background:#4ade80; box-shadow:0 0 0 5px rgba(74,222,128,.12); }
.gg-card { background:rgba(255,255,255,.92); border:1px solid #dce5ee; border-radius:18px; padding:18px 20px; box-shadow:0 5px 18px rgba(20,46,74,.045); height:100%; animation:rise .5s ease-out both; }
.gg-label { color:#64748b; font-size:12px; text-transform:uppercase; font-weight:700; letter-spacing:.08em; }
.gg-value { color:#101828; font-family:'Space Grotesk',sans-serif; font-size:27px; font-weight:700; margin-top:3px; }
.gg-note { color:#718096; font-size:12px; margin-top:3px; }
.gg-section { font-family:'Space Grotesk',sans-serif; font-size:22px; font-weight:700; letter-spacing:-.5px; margin:4px 0 3px; }
.gg-muted { color:#64748b; font-size:14px; }
.gg-stage { background:#fff; border:1px solid #dce5ee; border-radius:14px; padding:14px 15px; min-height:95px; position:relative; transition:all .25s ease; }
.gg-stage-active { border:1px solid #0eafc4; box-shadow:0 0 0 3px rgba(14,175,196,.12); transform:translateY(-2px); }
.gg-stage-title { font-weight:700; font-size:14px; color:#172033; }
.gg-stage-sub { color:#64748b; font-size:12px; margin-top:5px; }
.gg-kicker { color:#0e8fa1; font-size:11px; font-weight:800; letter-spacing:.1em; text-transform:uppercase; }
div.stButton > button { border-radius:12px; min-height:43px; font-weight:700; border:1px solid #d7e1ea; transition:transform .15s, box-shadow .15s; }
div.stButton > button:hover { transform:translateY(-1px); box-shadow:0 7px 16px rgba(13,39,68,.10); border-color:#0eafc4; }
div.stButton > button[kind="primary"] { background:#101a2d; color:#fff; border-color:#101a2d; }
[data-testid="stMetric"] { background:white; border:1px solid #dce5ee; padding:14px 17px; border-radius:16px; box-shadow:0 4px 14px rgba(20,46,74,.04); }
[data-testid="stMetricLabel"] { color:#64748b; }
hr { border-color:#dce5ee; }
@keyframes rise { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:translateY(0); } }
@keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:.45; } }
@media(max-width:700px){ .block-container{padding-left:1rem;padding-right:1rem;} .gg-hero{padding:22px;} .gg-card{padding:14px;} }
</style>
""", unsafe_allow_html=True)

FAULTS = ["Normal", "Line-to-Ground", "Line-to-Line", "Three-Phase", "Overload"]
LINES = ["N1-N2","N2-N3","N1-N4","N2-N5","N3-N6","N4-N5","N5-N6","N4-N7","N5-N8","N6-PV1","N7-N8","N8-PV1"]
FEATURES = ["voltage_sag_pu","current_multiple","phase_imbalance","frequency_deviation_hz","line_loading_pct","neighbor_anomaly","solar_output_mw","battery_soc_pct","sensor_noise","thermal_margin_pct","fault_duration_s"]
LOC_FEATURES = FEATURES + [f"segment_sensor_{i:02d}" for i in range(1,13)]
EDGES = [
    ("N1","N2","L01"),("N2","N3","L02"),("N1","N4","L03"),("N2","N5","L04"),
    ("N3","N6","L05"),("N4","N5","L06"),("N5","N6","L07"),("N4","N7","L08"),
    ("N5","N8","L09"),("N6","PV1","L10"),("N7","N8","L11"),("N8","PV1","L12")
]
LINE_TO_EDGE = {e[2]:e for e in EDGES}
LINE_LABEL_TO_ID = {label:f"L{i+1:02d}" for i,label in enumerate(LINES)}
NODE_POS = {"N1":(0,2),"N2":(1,2),"N3":(2,2),"N4":(0,1),"N5":(1,1),"N6":(2,1),"N7":(0,0),"N8":(1,0),"PV1":(2,0),"BAT":(.45,.35)}
DEFAULT_METRICS = None

def ensure_models():
    required = [MODELS/"fault_classifier.joblib", MODELS/"fault_localizer.joblib", MODELS/"model_metrics.json"]
    if not all(p.exists() for p in required):
        with st.spinner("Preparing synthetic dataset and training the ML models for first launch…"):
            result = subprocess.run([sys.executable, str(ROOT/"train_models.py")], capture_output=True, text=True)
        if result.returncode != 0:
            st.error("Model setup failed. Check the terminal output and install requirements.")
            st.code(result.stderr[-4000:])
            st.stop()
    return load(required[0]), load(required[1]), json.loads(required[2].read_text())

clf, localizer, metrics = ensure_models()

if "result" not in st.session_state: st.session_state.result = None
if "events" not in st.session_state: st.session_state.events = []
if "fault_active" not in st.session_state: st.session_state.fault_active = False
if "stage" not in st.session_state: st.session_state.stage = 0
if "selected_plan" not in st.session_state: st.session_state.selected_plan = "Plan B"
if "before_after" not in st.session_state: st.session_state.before_after = "After"

def event(msg, kind="INFO"):
    st.session_state.events.insert(0, {"time":pd.Timestamp.now().strftime("%H:%M:%S"),"kind":kind,"message":msg})
    st.session_state.events = st.session_state.events[:30]

def make_features(fault, line, solar, battery_on, seed=7):
    rng=np.random.default_rng(seed)
    params={
        "Normal":(.99,1.0,.03,0.0,62,.08,38,.8),
        "Line-to-Ground":(.72,2.25,.42,-.06,108,.55,4,5.0),
        "Line-to-Line":(.65,2.7,.62,-.08,118,.70,-3,5.0),
        "Three-Phase":(.48,3.5,.13,-.15,145,.90,-18,3.0),
        "Overload":(.88,1.75,.07,-.025,128,.30,-9,9.0)
    }
    v,i,imb,f,load,neigh,thermal,dur=params[fault]
    soc=74 if battery_on else 42
    d={
        "voltage_sag_pu":float(np.clip(rng.normal(v,.025),.1,1.15)),
        "current_multiple":float(np.clip(rng.normal(i,.08),.1,5)),
        "phase_imbalance":float(np.clip(abs(rng.normal(imb,.025)),0,1)),
        "frequency_deviation_hz":float(rng.normal(f,.015)),
        "line_loading_pct":float(np.clip(rng.normal(load,5),0,200)),
        "neighbor_anomaly":float(np.clip(rng.normal(neigh,.025),0,1.5)),
        "solar_output_mw":float(solar),
        "battery_soc_pct":float(soc),
        "sensor_noise":float(rng.uniform(.01,.06)),
        "thermal_margin_pct":float(rng.normal(thermal,4)),
        "fault_duration_s":float(rng.uniform(max(.1,dur*.5),dur*1.5))
    }
    locidx=LINES.index(line)
    for j in range(12):
        d[f"segment_sensor_{j+1:02d}"]=float((.8 if j==locidx else 0)+rng.normal(0,.06)+(d["neighbor_anomaly"]*.05))
    return d

def make_candidates(fault_line, solar, battery_on):
    # Illustrative candidates. Scores and load restoration are computed by transparent heuristics.
    alternatives = [
        {"name":"Plan A","path":["N5","N7","N8","N6"],"switches":4,"base_load":87,"loss":5.2,"voltage":.946,"thermal":91},
        {"name":"Plan B","path":["N5","N8","N6"],"switches":3,"base_load":94,"loss":2.1,"voltage":.982,"thermal":82},
        {"name":"Plan C","path":["N5","N7","N6"],"switches":5,"base_load":91,"loss":4.4,"voltage":.961,"thermal":108}
    ]
    # adapt solar and battery contribution modestly for the demo
    for a in alternatives:
        a["load_restored"]=min(99, max(40, a["base_load"] + (solar-2.0)*1.8 + (3 if battery_on else -1)))
        a["loss_pct"]=max(.8, a["loss"] + (2.0-solar)*.35 - (0.25 if battery_on else 0))
        a["voltage_pu"]=min(1.05, a["voltage"] + (solar-2)*.004 + (.008 if battery_on else -.006))
        a["thermal_pct"]=a["thermal"] + max(0,solar-2)*2.0 - (4 if battery_on else 0)
        a["score"]=round(max(0,min(1, .42*a["load_restored"]/100 + .24*(1-a["loss_pct"]/15) + .18*a["voltage_pu"] + .16*(1-a["switches"]/8))),3)
        reasons=[]
        if a["voltage_pu"] < .90 or a["voltage_pu"] > 1.05: reasons.append("Voltage outside configured 0.90–1.05 pu range")
        if a["thermal_pct"] > 100: reasons.append("Thermal loading exceeds 100% limit")
        if a["load_restored"] < 70: reasons.append("Restored load below 70% target")
        a["valid"] = not reasons
        a["reasons"] = reasons or ["Passed configured demo checks"]
        a["decision"] = "RECOMMEND" if a["valid"] and a["score"]>=.78 else ("REVIEW" if a["valid"] else "REJECT")
    return alternatives

def run_pipeline(fault, line, solar, battery_on):
    row=make_features(fault,line,solar,battery_on,seed=int(pd.Timestamp.now().timestamp())%100000)
    X=pd.DataFrame([[row[f] for f in FEATURES]],columns=FEATURES)
    predicted=clf.predict(X)[0]
    probs=clf.predict_proba(X)[0]
    confidence=float(max(probs))
    LX=pd.DataFrame([[row[f] for f in LOC_FEATURES]],columns=LOC_FEATURES)
    locpred=localizer.predict(LX)[0]
    locprobs=localizer.predict_proba(LX)[0]
    locconf=float(max(locprobs))
    candidates=make_candidates(line,solar,battery_on)
    candidates=sorted(candidates,key=lambda x:(x["valid"],x["score"]),reverse=True)
    best=next((c for c in candidates if c["valid"]),candidates[0])
    return {"input_fault":fault,"input_line":line,"predicted_fault":predicted,"detection_confidence":confidence,
            "fault_probabilities":{k:float(v) for k,v in zip(clf.classes_,probs)},
            "estimated_line":locpred,"localization_confidence":locconf,"features":row,"candidates":candidates,"best":best,
            "solar":solar,"battery":battery_on,"timestamp":pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}

def network_figure(result=None, after=False, selected=None):
    fig=go.Figure()
    fault_line=(result or {}).get("input_line", "N5-N6")
    best=(result or {}).get("best",{})
    recovery=set()
    if after and best:
        path=best.get("path",[])
        for a,b in zip(path[:-1],path[1:]): recovery.add(frozenset([a,b]))
    # lines
    for a,b,lid in EDGES:
        is_fault = result is not None and result.get("input_line")==f"{a}-{b}"
        is_recovery = frozenset([a,b]) in recovery
        col = "#ef4444" if (is_fault and not after) else "#0eafc4" if is_recovery else "#a8b8c8"
        dash = "dash" if is_fault and after else "solid"
        width=5 if is_recovery or is_fault else 3
        xa,ya=NODE_POS[a]; xb,yb=NODE_POS[b]
        fig.add_trace(go.Scatter(x=[xa,xb],y=[ya,yb],mode="lines",line=dict(color=col,width=width,dash=dash),
            hoverinfo="text",text=[f"{lid}: {a}–{b}",f"{lid}: {a}–{b}"],showlegend=False))
        # midpoint line label
        fig.add_annotation(x=(xa+xb)/2,y=(ya+yb)/2+.08,text=lid,font=dict(size=9,color="#718096"),showarrow=False,bgcolor="rgba(255,255,255,.7)")
    # nodes
    nx=[]; ny=[]; labels=[]; colors=[]; sizes=[]
    for node,(x,y) in NODE_POS.items():
        if node=="BAT": continue
        nx.append(x);ny.append(y);labels.append(node)
        if result and node in fault_line.split("-") and not after: colors.append("#ef4444")
        elif after and best and node in best.get("path",[]): colors.append("#0eafc4")
        else: colors.append("#279b53")
        sizes.append(25 if node not in ["PV1"] else 22)
    fig.add_trace(go.Scatter(x=nx,y=ny,mode="markers+text",text=labels,textposition="bottom center",
        marker=dict(size=sizes,color=colors,line=dict(color="white",width=2)),textfont=dict(size=12,color="#475569"),
        hovertemplate="%{text}<extra>Simulated bus</extra>",showlegend=False))
    # PV and battery
    for node,color,symbol in [("PV1","#22a06b","square"),("BAT","#f59e0b","square")]:
        x,y=NODE_POS[node]
        fig.add_trace(go.Scatter(x=[x],y=[y],mode="markers+text",text=[node],textposition="bottom center",
            marker=dict(size=22,color=color,symbol=symbol,line=dict(color="white",width=2)),textfont=dict(size=11,color="#475569"),showlegend=False))
    fig.update_layout(height=400,margin=dict(l=15,r=15,t=10,b=25),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#fbfdff",
        xaxis=dict(visible=False,range=[-.35,2.35]),yaxis=dict(visible=False,range=[-.4,2.45],scaleanchor="x",scaleratio=1),
        showlegend=False,dragmode=False)
    return fig

def metric_card(label,value,note,emoji):
    st.markdown(f"""<div class="gg-card"><div style="display:flex;gap:12px;align-items:center"><div style="font-size:22px;background:#eef3f8;border-radius:12px;padding:10px">{emoji}</div><div><div class="gg-label">{label}</div><div class="gg-value">{value}</div><div class="gg-note">{note}</div></div></div></div>""",unsafe_allow_html=True)

# Sidebar controls
with st.sidebar:
    st.markdown("## ⚡ GridGuard AI")
    st.caption("SIMULATED DISTRIBUTION GRID")
    st.divider()
    page=st.radio("NAVIGATION",["Live Dashboard","Model Performance","Dataset Explorer","About & Limitations"],label_visibility="visible")
    st.divider()
    st.markdown("### Simulation controls")
    fault_line=st.selectbox("Fault location",LINES,index=6)
    fault_type=st.selectbox("Fault type",FAULTS,index=1)
    solar=st.slider("Solar generation (MW)",0.0,3.0,2.4,.1)
    battery=st.toggle("Battery support enabled",value=True)
    st.caption("All controls affect a local simulation only.")
    inject=st.button("⚡ Inject Fault",type="primary",use_container_width=True)
    full_demo=st.button("▶ Run Full Self-Healing Demo",use_container_width=True)
    reset=st.button("↺ Reset simulation",use_container_width=True)
    if inject or full_demo:
        st.session_state.fault_active=True
        st.session_state.result=run_pipeline(fault_type,fault_line,solar,battery)
        st.session_state.stage=6 if full_demo else 1
        event(f"Fault injected: {fault_type} on {fault_line}","FAULT")
        event(f"ML predicted {st.session_state.result['predicted_fault']}","MODEL")
        event(f"Localization estimate: {st.session_state.result['estimated_line']}","MODEL")
        event(f"{st.session_state.result['best']['name']} selected as top valid candidate","PLAN")
        if full_demo:
            event("Candidate validation completed; simulated recovery path displayed","RECOVERY")
        else:
            event("Fault detection and localization complete; inspect candidate plans","INFO")
    if reset:
        st.session_state.result=None; st.session_state.fault_active=False; st.session_state.stage=0
        st.session_state.events=[]; event("Simulation reset to healthy baseline","INFO")

result=st.session_state.result
# Header
st.markdown("""
<div class="gg-hero">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:18px;flex-wrap:wrap">
    <div><div class="gg-brand">🛡️ GridGuard <span style="color:#36d4e4">AI</span></div>
      <div class="gg-sub">AI-Driven Self-Healing Smart Grid · Interactive web simulator</div>
      <div style="margin-top:18px"><span class="gg-pill"><span class="gg-dot"></span> SIMULATION ENVIRONMENT</span></div>
    </div>
    <div style="min-width:190px;padding:14px 18px;border:1px solid rgba(255,255,255,.15);border-radius:16px;background:rgba(255,255,255,.06)">
      <div style="font-size:11px;color:#b9ccdd;font-weight:700;letter-spacing:.1em">SYSTEM STATUS</div>
      <div style="font-size:22px;font-weight:700;margin-top:4px">""" + ("FAULT SCENARIO" if result else "GRID HEALTHY") + """</div>
      <div style="font-size:12px;color:#c5d6e5;margin-top:3px">""" + ("Recovery plan ready for review" if result else "No active fault injected") + """</div>
    </div>
  </div>
</div>
""",unsafe_allow_html=True)
st.write("")

if page=="Live Dashboard":
    st.markdown('<div class="gg-section">Self-Healing Process</div><div class="gg-muted">Click a stage to inspect the simulated AI pipeline. Run the demo to animate the complete workflow.</div>',unsafe_allow_html=True)
    stage_names=[("Fault","Abnormal event"),("Detection","ML predicts fault class"),("Localization","Model estimates line"),("Isolation","Faulted section marked"),("Optimization","Candidates scored"),("Validation","Constraints screened"),("Reconfiguration","Alternate path highlighted"),("Power Restored","Simulated state updated")]
    cols=st.columns(4)
    for i,(name,sub) in enumerate(stage_names):
        with cols[i%4]:
            active=result is not None and i <= st.session_state.stage
            st.markdown(f'<div class="gg-stage {"gg-stage-active" if active else ""}"><div class="gg-kicker">STAGE {i+1:02d}</div><div class="gg-stage-title">{name}</div><div class="gg-stage-sub">{sub}</div></div>',unsafe_allow_html=True)
            if st.button("Inspect",key=f"stage_{i}",use_container_width=True):
                st.session_state.stage=i
                st.session_state.inspect_stage=i
    if result and "inspect_stage" in st.session_state:
        i=st.session_state.inspect_stage
        descriptions=[
            f"Input event: {result['input_fault']} on {result['input_line']}. All values are simulated.",
            f"Classifier prediction: {result['predicted_fault']} with {result['detection_confidence']:.1%} model confidence.",
            f"Localization prediction: {result['estimated_line']} with {result['localization_confidence']:.1%} model confidence.",
            "The visualization marks the selected faulted section as isolated for the recovery-planning demonstration.",
            f"{len(result['candidates'])} candidate plans generated and ranked by a transparent demo score.",
            f"Best plan: {result['best']['name']}. Validation reasons: " + "; ".join(result['best']['reasons']),
            "The alternate path is highlighted on the after-recovery network diagram.",
            "This is a simulated recovery outcome, not a command sent to physical equipment."
        ]
        st.info(descriptions[i])
    st.write("")
    # KPI row
    if result:
        gridhealth = 98.7 if result["best"]["valid"] else 82.4
        restored = result["best"]["load_restored"]
        conf = result["detection_confidence"]
        active_nodes = "9 / 9"
        status = result["best"]["decision"]
    else:
        gridhealth=100.0; restored=0; conf=0; active_nodes="9 / 9"; status="HEALTHY"
    c=st.columns(4)
    with c[0]: metric_card("Grid Health",f"{gridhealth:.1f}%","simulated index","〽️")
    with c[1]: metric_card("Active Buses",active_nodes,"9 network buses","▦")
    with c[2]: metric_card("Solar Output",f"{solar:.1f} MW",f"of 3.0 MW capacity","☀️")
    with c[3]: metric_card("Decision Status",status,"operator review required","🛡️")
    st.write("")
    left,right=st.columns([1.35,1],gap="large")
    with left:
        st.markdown('<div class="gg-section">Distribution Network</div><div class="gg-muted">Select a line in the sidebar, then inject a fault. Red = fault; cyan = candidate recovery path.</div>',unsafe_allow_html=True)
        view=st.radio("Grid view",["Before event","After recovery"],horizontal=True,label_visibility="collapsed",key="grid_view")
        st.plotly_chart(network_figure(result,after=(view=="After recovery" and result is not None)),use_container_width=True,config={"displayModeBar":False})
        st.caption("Illustrative 9-bus distribution topology with a solar PV node and battery. Not an electrical power-flow model.")
    with right:
        st.markdown('<div class="gg-section">AI Recovery Agent</div><div class="gg-muted">Outputs from trained synthetic-data models and demo planning rules.</div>',unsafe_allow_html=True)
        if result:
            st.markdown(f"""<div class="gg-card">
              <div class="gg-label">STATUS</div><div class="gg-value" style="font-size:22px;color:#16834b">READY FOR REVIEW</div>
              <hr><div class="gg-label">Predicted fault</div><div style="font-size:18px;font-weight:700">{result['predicted_fault']}</div>
              <div class="gg-note">Confidence: {result['detection_confidence']:.1%}</div>
              <hr><div class="gg-label">Estimated line</div><div style="font-size:18px;font-weight:700">{result['estimated_line']}</div>
              <div class="gg-note">Localization model confidence: {result['localization_confidence']:.1%}</div>
              <hr><div class="gg-label">Best candidate</div><div style="font-size:18px;font-weight:700">{result['best']['name']} · {result['best']['decision']}</div>
              <div class="gg-note">Illustrative candidate score: {result['best']['score']:.3f}</div>
            </div>""",unsafe_allow_html=True)
            st.write("")
            st.progress(min(1.0,st.session_state.stage/7),text=f"Self-healing workflow progress · {int(min(1,st.session_state.stage/7)*100)}%")
        else:
            st.markdown('<div class="gg-card" style="text-align:center;padding:42px 20px"><div style="font-size:38px">〽️</div><div style="font-weight:700;font-size:17px;margin-top:10px">Awaiting simulation</div><div class="gg-muted" style="margin-top:8px">Inject a fault or run the full demo to train the workflow on a new event.</div></div>',unsafe_allow_html=True)
            st.write("")
            st.info("Try Line-to-Ground at N5–N6, then click **Run Full Self-Healing Demo** in the sidebar.")
    st.write("")
    st.markdown('<div class="gg-section">Candidate Recovery Configurations</div><div class="gg-muted">Candidates are illustrative switching paths ranked by a transparent heuristic and checked against configured demo limits.</div>',unsafe_allow_html=True)
    if result:
        plans=result["candidates"]
        pcols=st.columns(3)
        for i,p in enumerate(plans):
            with pcols[i]:
                accent="#16834b" if p["valid"] else "#dc2626"
                st.markdown(f"""<div class="gg-card" style="border-top:4px solid {accent}">
                <div style="display:flex;justify-content:space-between;gap:6px;align-items:center"><div style="font-size:18px;font-weight:700">{p['name']}</div><div style="font-size:10px;font-weight:800;color:{accent};background:#f1f5f9;border-radius:99px;padding:5px 8px">{p['decision']}</div></div>
                <div class="gg-note" style="margin-top:8px">Path: {' → '.join(p['path'])}</div>
                <div style="font-size:24px;font-weight:700;margin-top:14px">{p['load_restored']:.1f}% <span style="font-size:12px;color:#64748b;font-weight:500">load restored*</span></div>
                <div style="margin-top:10px;font-size:13px;color:#475569">Loss proxy <b>{p['loss_pct']:.1f}%</b> · Voltage proxy <b>{p['voltage_pu']:.3f} pu</b></div>
                <div style="margin-top:6px;font-size:13px;color:#475569">Thermal proxy <b>{p['thermal_pct']:.1f}%</b> · Switch ops <b>{p['switches']}</b></div>
                <div style="margin-top:12px;font-size:12px;color:{accent}">{'; '.join(p['reasons'])}</div></div>""",unsafe_allow_html=True)
                if st.button(f"Preview {p['name']}",key=f"preview_{p['name']}",use_container_width=True):
                    st.session_state.selected_plan=p["name"]
                    st.session_state.result["best"]=p
                    event(f"{p['name']} selected for preview","PLAN")
                    st.rerun()
        st.caption("*Load-restoration, loss, voltage, thermal, and score values are simulated illustrative proxies—not measured results.")
    else:
        st.markdown('<div class="gg-card" style="text-align:center;color:#64748b">Recovery configurations appear after fault injection and model inference.</div>',unsafe_allow_html=True)
    st.write("")
    # PV and battery cards
    p1,p2=st.columns(2)
    with p1:
        st.markdown(f"""<div class="gg-card"><div style="font-size:18px;font-weight:700">☀️ Solar PV (PV1)</div><div style="display:flex;justify-content:space-between;margin-top:16px;color:#64748b">Generation <b style="color:#111827">{solar:.1f} MW</b></div><div style="display:flex;justify-content:space-between;margin-top:10px;color:#64748b">Capacity <b style="color:#111827">3.0 MW</b></div><div style="display:flex;justify-content:space-between;margin-top:10px;color:#64748b">Status <b style="color:#16834b">● ONLINE</b></div></div>""",unsafe_allow_html=True)
    with p2:
        st.markdown(f"""<div class="gg-card"><div style="font-size:18px;font-weight:700">🔋 Battery (BAT)</div><div style="display:flex;justify-content:space-between;margin-top:16px;color:#64748b">State of charge <b style="color:#111827">{74 if battery else 42}%</b></div><div style="display:flex;justify-content:space-between;margin-top:10px;color:#64748b">Power support <b style="color:{'#16834b' if battery else '#64748b'}">{'+0.8 MW AVAILABLE' if battery else 'DISABLED'}</b></div><div class="gg-note" style="margin-top:12px">Change support in the sidebar and rerun the simulation to update candidate scoring.</div></div>""",unsafe_allow_html=True)
    st.write("")
    ev1,ev2=st.columns([1.2,1])
    with ev1:
        st.markdown('<div class="gg-section">Live Event Log</div>',unsafe_allow_html=True)
        if st.session_state.events:
            log=pd.DataFrame(st.session_state.events)
            st.dataframe(log,use_container_width=True,hide_index=True,height=220)
        else:
            st.markdown('<div class="gg-card" style="font-family:monospace;color:#64748b">No events yet. Click “Run Full Self-Healing Demo” or “Inject Fault” to begin.</div>',unsafe_allow_html=True)
    with ev2:
        st.markdown('<div class="gg-section">System Intelligence</div>',unsafe_allow_html=True)
        st.markdown('<div class="gg-card"><b>ML Fault Detector</b><div class="gg-note">Random Forest predicts fault category from simulated measurements.</div><br><b>Fault Localizer</b><div class="gg-note">Random Forest estimates the faulted segment from simulated sensor signatures.</div><br><b>Recovery Planner</b><div class="gg-note">Ranks candidate paths and applies explicit demo constraint checks.</div></div>',unsafe_allow_html=True)
        st.caption("No GNN, RL agent, live grid connection, or validated AC load-flow solver is implemented in this version.")

elif page=="Model Performance":
    st.markdown('<div class="gg-section">Model Performance</div><div class="gg-muted">Measured on held-out synthetic data generated by this project. These metrics do not represent real-grid performance.</div>',unsafe_allow_html=True)
    a,b=st.columns(2)
    for col,key,title in [(a,"fault_classifier","Fault Classification"),(b,"fault_localizer","Fault Localization")]:
        m=metrics[key]
        with col:
            st.markdown(f'<div class="gg-card"><div style="font-size:18px;font-weight:700">{title}</div><div class="gg-note">{m["training_samples"]:,} train samples · {m["test_samples"]:,} test samples</div></div>',unsafe_allow_html=True)
            mm=st.columns(4)
            for c,(label,val) in zip(mm,[("Accuracy",m["accuracy"]),("Precision",m["precision_macro"]),("Recall",m["recall_macro"]),("Macro F1",m["f1_macro"])]):
                c.metric(label,f"{val:.1%}")
            cm=np.array(m["confusion_matrix"])
            fig=go.Figure(data=go.Heatmap(z=cm,x=m["labels"],y=m["labels"],colorscale=[[0,"#eff6ff"],[.5,"#67c9d6"],[1,"#123a5b"]],text=cm,texttemplate="%{text}",hovertemplate="Actual %{y}<br>Predicted %{x}<br>Count %{z}<extra></extra>"))
            fig.update_layout(title="Confusion matrix",height=390,margin=dict(l=10,r=10,t=45,b=10),paper_bgcolor="rgba(0,0,0,0)",font=dict(color="#334155"))
            fig.update_xaxes(title="Predicted"); fig.update_yaxes(title="Actual")
            st.plotly_chart(fig,use_container_width=True)
    st.markdown('<div class="gg-section">Classification report</div>',unsafe_allow_html=True)
    model_choice=st.selectbox("Select model report",["Fault classifier","Fault localizer"])
    key="fault_classifier" if model_choice=="Fault classifier" else "fault_localizer"
    report=metrics[key]["classification_report"]
    rows=[]
    for label,vals in report.items():
        if isinstance(vals,dict) and "precision" in vals:
            rows.append({"Class":label,"Precision":vals["precision"],"Recall":vals["recall"],"F1-score":vals["f1-score"],"Support":vals["support"]})
    st.dataframe(pd.DataFrame(rows).style.format({"Precision":"{:.3f}","Recall":"{:.3f}","F1-score":"{:.3f}"}),use_container_width=True,hide_index=True)
    st.download_button("Download model metrics JSON",data=json.dumps(metrics,indent=2),file_name="gridguard_model_metrics.json",mime="application/json")

elif page=="Dataset Explorer":
    st.markdown('<div class="gg-section">Synthetic Dataset Explorer</div><div class="gg-muted">Explore the generated sample records used to train the two ML models.</div>',unsafe_allow_html=True)
    data_path=DATA/"grid_fault_dataset.csv"
    if not data_path.exists():
        st.warning("Dataset is created on first launch. Restart the app after training.")
    else:
        df=pd.read_csv(data_path)
        a,b,c=st.columns(3)
        a.metric("Rows",f"{len(df):,}"); b.metric("Features",len(FEATURES)); c.metric("Fault classes",df["fault_type"].nunique())
        selected=st.multiselect("Filter fault types",FAULTS,default=FAULTS)
        filtered=df[df["fault_type"].isin(selected)]
        st.dataframe(filtered.head(500),use_container_width=True,height=420)
        st.caption("Showing up to 500 filtered rows for browser performance. Download the complete CSV below.")
        st.download_button("Download full synthetic dataset CSV",data=data_path.read_bytes(),file_name="grid_fault_dataset.csv",mime="text/csv")
        left,right=st.columns(2)
        with left:
            cnt=df["fault_type"].value_counts().rename_axis("Fault type").reset_index(name="Samples")
            fig=go.Figure(go.Bar(x=cnt["Fault type"],y=cnt["Samples"],marker_color="#0eafc4"))
            fig.update_layout(title="Class distribution",height=330,margin=dict(l=10,r=10,t=45,b=10),paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig,use_container_width=True)
        with right:
            fig=go.Figure(go.Histogram(x=df["voltage_sag_pu"],nbinsx=30,marker_color="#249b52"))
            fig.update_layout(title="Voltage measurement distribution",height=330,margin=dict(l=10,r=10,t=45,b=10),paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig,use_container_width=True)

else:
    st.markdown('<div class="gg-section">About GridGuard AI</div>',unsafe_allow_html=True)
    st.markdown("""
    <div class="gg-card">
    <h3>Project objective</h3>
    GridGuard AI explores a software workflow for simulated fault detection, line localization, recovery candidate comparison, and constraint-aware decision support.
    <h3>Implemented in this package</h3>
    <ul>
      <li>Reproducible synthetic dataset generation and two trained Random Forest models.</li>
      <li>Held-out evaluation metrics, confusion matrices and per-class reports.</li>
      <li>Interactive grid topology, fault injection, solar and battery controls, recovery-plan scoring, event history, and before/after views.</li>
      <li>Local-first application that does not require API keys.</li>
    </ul>
    <h3>Not implemented / not validated</h3>
    <ul>
      <li>No real SCADA/sensor connectivity, relay protection logic, or physical switch control.</li>
      <li>No trained graph neural network (GNN) or reinforcement-learning agent in this version.</li>
      <li>Grid physics and constraint checks are simplified proxies, not validated power-flow calculations.</li>
      <li>Synthetic test metrics must not be presented as real-world results.</li>
    </ul>
    <h3>Research-grade next step</h3>
    Integrate a standard feeder with a validated power-flow engine such as pandapower, generate fault cases from the feeder model, compare against transparent baselines, and ask a power-systems engineer to review the assumptions.
    </div>
    """,unsafe_allow_html=True)
    st.warning("Simulation only. Never use this app to make or execute switching decisions on a real electrical grid.")
