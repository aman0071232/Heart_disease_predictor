import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="AI Heart Predictor", page_icon="❤️", layout="wide")

# =========================
# ADVANCED CSS (PRO LEVEL 🔥)
# =========================
st.markdown("""
<style>

/* BACKGROUND */
.main {
    background: linear-gradient(120deg, #020617, #0f172a, #020617);
}

/* TITLE ANIMATION */
@keyframes fadeIn {
    from {opacity: 0; transform: translateY(-20px);}
    to {opacity: 1; transform: translateY(0);}
}

/* GLASS CARD */
.glass {
    background: rgba(255,255,255,0.05);
    border-radius: 20px;
    padding: 25px;
    backdrop-filter: blur(15px);
    border: 1px solid rgba(255,255,255,0.1);
    transition: 0.3s;
}
.glass:hover {
    transform: translateY(-5px) scale(1.01);
    box-shadow: 0px 0px 25px rgba(255,75,75,0.2);
}

/* BUTTON */
.stButton>button {
    background: linear-gradient(45deg,#ff4b4b,#ff6b6b);
    color: white;
    border-radius: 15px;
    height: 3em;
    font-size: 18px;
    transition: 0.3s;
}
.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 20px rgba(255,75,75,0.6);
}

/* METRIC CARD */
.metric-card {
    text-align:center;
    padding:15px;
    border-radius:15px;
    background: rgba(255,255,255,0.05);
}

/* TITLE */
.title {
    font-size: 50px;
    font-weight: bold;
    text-align:center;
    animation: fadeIn 1s ease-in-out;
}

/* SUBTITLE */
.subtitle {
    text-align:center;
    color: #9ca3af;
    margin-bottom: 20px;
}

</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
model = joblib.load("KNN_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

# =========================
# HERO SECTION
# =========================
st.markdown('<div class="title">❤️ AI Heart Risk Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Advanced Machine Learning powered diagnosis system</div>', unsafe_allow_html=True)

# =========================
# INPUT SECTION (GRID STYLE)
# =========================
st.markdown("### 🧾 Enter Patient Details")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 18, 100, 40)
    sex = st.selectbox("Sex", ["M", "F"])
    chest_pain = st.selectbox("Chest Pain", ["ATA", "NAP", "TA", "ASY"])

with col2:
    resting_bp = st.number_input("Resting BP", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol", 100, 600, 200)
    fasting_bs = st.selectbox("Fasting Sugar > 120", [0, 1])

with col3:
    max_hr = st.slider("Max Heart Rate", 60, 220, 150)
    oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)
    exercise_angina = st.selectbox("Exercise Angina", ["Y", "N"])

resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# =========================
# PREDICT BUTTON
# =========================
predict_btn = st.button("🚀 Analyze Heart Risk")

# =========================
# PREDICTION
# =========================
if predict_btn:

    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]
    scaled_input = scaler.transform(input_df)

    prediction = model.predict(scaled_input)[0]
    prob = model.predict_proba(scaled_input)[0]

    st.markdown("---")

    # =========================
    # RESULT CARD
    # =========================
    colA, colB, colC = st.columns(3)

    colA.metric("Risk %", f"{prob[1]*100:.1f}%")
    colB.metric("Heart Rate", max_hr)
    colC.metric("Cholesterol", cholesterol)

    if prediction == 1:
        st.error("⚠️ HIGH RISK DETECTED")
    else:
        st.success("✅ LOW RISK")

    # =========================
    # DONUT CHART (ANIMATED LOOK)
    # =========================
    fig = go.Figure(data=[go.Pie(
        labels=["Safe", "Risk"],
        values=[prob[0], prob[1]],
        hole=0.7
    )])

    fig.update_layout(
        title="AI Risk Analysis",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white")
    )

    st.plotly_chart(fig, use_container_width=True)

    # =========================
    # SMOOTH COMPARISON CHART
    # =========================
    df_compare = pd.DataFrame({
        "Feature": ["Age","BP","Chol","HR","Oldpeak"],
        "Normal": [40,120,200,150,1],
        "You": [age,resting_bp,cholesterol,max_hr,oldpeak]
    })

    fig2 = px.line(df_compare, x="Feature", y=["Normal","You"], markers=True)

    fig2.update_traces(line=dict(width=4))

    fig2.update_layout(
        title="Health vs Ideal",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white")
    )

    st.plotly_chart(fig2, use_container_width=True)

    # =========================
    # MODERN BAR CHART
    # =========================
    df_bar = pd.DataFrame({
        "Metric": ["Cholesterol","BP","MaxHR"],
        "Value": [cholesterol, resting_bp, max_hr]
    })

    fig3 = px.bar(df_bar, x="Metric", y="Value", text="Value")

    fig3.update_layout(
        title="Key Indicators",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white")
    )

    st.plotly_chart(fig3, use_container_width=True)