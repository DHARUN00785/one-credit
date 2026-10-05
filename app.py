import streamlit as st
import tensorflow as tf
import numpy as np

# ─────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Patient Readmission Predictor",
    page_icon="🏥",
    layout="wide"
)

# ─────────────────────────────────────────────
# CSS Styling
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .main { background-color: #f0f4f8; }

    .stButton > button {
        width: 100%;
        border-radius: 8px;
        height: 3.2em;
        background: linear-gradient(135deg, #004a99, #0077cc);
        color: white;
        font-weight: 700;
        font-size: 1.05em;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,74,153,0.3);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,74,153,0.4);
    }

    /* Risk result card */
    .risk-card {
        border-radius: 16px;
        padding: 28px 32px;
        text-align: center;
        margin-top: 10px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
        animation: fadeIn 0.5s ease;
    }
    .risk-high {
        background: linear-gradient(135deg, #e63946, #c1121f);
        color: white;
    }
    .risk-moderate {
        background: linear-gradient(135deg, #f4a261, #e76f51);
        color: white;
    }
    .risk-low {
        background: linear-gradient(135deg, #2a9d8f, #21867a);
        color: white;
    }
    .risk-card h2 { font-size: 2.2em; margin: 0; font-weight: 700; }
    .risk-card h3 { font-size: 1.3em; margin: 6px 0 0 0; font-weight: 400; opacity: 0.9; }

    /* Gauge bar */
    .gauge-container {
        background: #e2e8f0;
        border-radius: 999px;
        height: 18px;
        overflow: hidden;
        margin: 14px 0 6px 0;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
    }
    .gauge-fill {
        height: 100%;
        border-radius: 999px;
        transition: width 0.6s ease;
    }

    /* Section headers */
    .section-label {
        font-size: 0.78em;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: #64748b;
        margin-bottom: 4px;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to   { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Model Loading
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('readmission_model (1).keras')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown("## 🏥 Patient Readmission Risk Assessment")
st.markdown("Enter patient clinical data below and click **Predict** to assess the likelihood of hospital readmission within 30 days.")
st.divider()

# ─────────────────────────────────────────────
# Input Form
# ─────────────────────────────────────────────
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<p class="section-label">👤 Patient Demographics</p>', unsafe_allow_html=True)
    gender = st.radio("Gender", ["Female", "Male"], horizontal=True)
    gender_val = 1 if gender == "Male" else 0

    age_options = [
        "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
        "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"
    ]
    age_opt = st.selectbox("Age Range", age_options, index=6)
    age_val = age_options.index(age_opt)

    st.markdown('<p class="section-label">🏥 Current Hospitalisation</p>', unsafe_allow_html=True)
    time_in_hosp = st.slider("Time in Hospital (Days)", 1, 14, 3)
    num_diagnoses = st.slider("Number of Diagnoses", 1, 16, 5)

with col2:
    st.markdown('<p class="section-label">🔬 Clinical Procedures & Medications</p>', unsafe_allow_html=True)
    num_lab_procs = st.slider("Number of Lab Procedures", 1, 130, 40)
    num_procs = st.slider("Number of Other Procedures", 0, 10, 1)
    num_meds = st.slider("Number of Medications", 1, 80, 15)

    st.markdown('<p class="section-label">📈 Prior Healthcare Utilisation (Past Year)</p>', unsafe_allow_html=True)
    num_outpatient = st.slider("Outpatient Visits", 0, 42, 0)
    num_emergency  = st.slider("Emergency Visits",  0, 76, 0)
    num_inpatient  = st.slider("Inpatient Visits",  0, 21, 0)

# ─────────────────────────────────────────────
# StandardScaler parameters (from diabetic_data.csv training set)
# The model was trained on StandardScaler-normalized features.
# ─────────────────────────────────────────────
FEATURE_MEAN = np.array([0.462, 6.088, 4.396, 43.096, 1.340, 16.022, 0.369, 0.198, 0.636, 7.423], dtype=np.float32)
FEATURE_STD  = np.array([0.499, 1.604, 2.985, 19.671, 1.705, 8.128, 1.267, 0.931, 1.263, 1.938], dtype=np.float32)

# ─────────────────────────────────────────────
# Build feature vector & apply StandardScaler
# Feature order: gender, age, time_in_hospital, num_lab_procedures, num_procedures,
#                num_medications, number_outpatient, number_emergency,
#                number_inpatient, number_diagnoses
# ─────────────────────────────────────────────
def get_prediction_array():
    raw = np.array([[
        gender_val, age_val, time_in_hosp, num_lab_procs, num_procs,
        num_meds, num_outpatient, num_emergency, num_inpatient, num_diagnoses
    ]], dtype=np.float32)
    # Apply the same StandardScaler used during training
    scaled = (raw - FEATURE_MEAN) / FEATURE_STD
    return scaled

# ─────────────────────────────────────────────
# Risk Tier Configuration
# With StandardScaler, model outputs span a meaningful range:
#   Extreme-high-risk patients: ~20-25%
#   Medium-risk patients:       ~5-10%
#   Low-risk patients:          ~0.5-4%
# ─────────────────────────────────────────────
def classify_risk(p: float):
    """Return (tier_label, tier_class, explanation)."""
    if p >= 0.15:
        return "🔴 High Risk", "risk-high", (
            "This patient has a significantly elevated readmission risk. "
            "Consider enhanced discharge planning and close follow-up."
        )
    elif p >= 0.06:
        return "🟠 Moderate Risk", "risk-moderate", (
            "This patient shows moderate readmission risk indicators. "
            "Ensure a clear follow-up plan before discharge."
        )
    else:
        return "🟢 Low Risk", "risk-low", (
            "This patient currently shows low readmission risk indicators. "
            "Standard discharge protocols are appropriate."
        )

# ─────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────
st.divider()
predict_col, _ = st.columns([1, 2])
with predict_col:
    run_pred = st.button("🔍 Predict Readmission Risk", use_container_width=True)

if run_pred:
    with st.spinner("Analysing patient data…"):
        input_data = get_prediction_array()
        raw_prob   = float(model.predict(input_data, verbose=0)[0][0])

    label, css_class, advice = classify_risk(raw_prob)
    pct = raw_prob * 100

    # Gauge colour
    if raw_prob >= 0.15:
        gauge_colour = "#e63946"
    elif raw_prob >= 0.06:
        gauge_colour = "#f4a261"
    else:
        gauge_colour = "#2a9d8f"

    # Clamp display bar to max 100 %
    bar_width = min(pct * 4, 100)   # scale for visibility

    st.markdown(f"""
    <div class="risk-card {css_class}">
        <h3>Readmission Probability</h3>
        <h2>{pct:.1f}%</h2>
        <h3>{label}</h3>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="gauge-container">
        <div class="gauge-fill" style="width:{bar_width:.1f}%; background:{gauge_colour};"></div>
    </div>
    <p style="text-align:center; color:#64748b; font-size:0.85em;">
        Risk scale — higher bar indicates greater predicted readmission risk
    </p>
    """, unsafe_allow_html=True)

    if raw_prob >= 0.15:
        st.error(f"⚠️ {advice}")
    elif raw_prob >= 0.06:
        st.warning(f"⚡ {advice}")
    else:
        st.success(f"✅ {advice}")

    # Feature summary
    with st.expander("📋 Patient Input Summary"):
        summary_data = {
            "Feature": [
                "Gender", "Age Range", "Time in Hospital",
                "Lab Procedures", "Other Procedures", "Medications",
                "Outpatient Visits", "Emergency Visits", "Inpatient Visits",
                "Number of Diagnoses"
            ],
            "Value": [
                gender, age_opt, f"{time_in_hosp} days",
                num_lab_procs, num_procs, num_meds,
                num_outpatient, num_emergency, num_inpatient, num_diagnoses
            ]
        }
        import pandas as pd
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

st.caption("⚕️ Note: This tool is a clinical decision-support aid. Clinician judgement should always prevail.")
