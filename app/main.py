# ============================================
# SMARTGUARD AI — PHASE 7
# STREAMLIT DASHBOARD
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ---- PAGE CONFIG ----
st.set_page_config(
    page_title="SmartGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- PATHS ----
MODEL_PATH    = r'D:\smarguard ai\ml_model\smartguard_model.pkl'
SCALER_PATH   = r'D:\smarguard ai\ml_model\scaler.pkl'
ENCODER_PATH  = r'D:\smarguard ai\ml_model\label_encoder.pkl'
FEATURES_PATH = r'D:\smarguard ai\ml_model\feature_names.pkl'
LOG_FILE      = r'D:\smarguard ai\logs\detections.csv'
FLOWS_CSV     = r'D:\smarguard ai\sensor\flows.csv'

# ---- CUSTOM CSS ----
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stApp { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252a40);
        border: 1px solid #3d4166;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .alert-box {
        background-color: #2d1b1b;
        border-left: 4px solid #ff4444;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .normal-box {
        background-color: #1b2d1b;
        border-left: 4px solid #44ff44;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# ---- LOAD MODEL ----
@st.cache_resource
def load_model():
    try:
        model    = joblib.load(MODEL_PATH)
        scaler   = joblib.load(SCALER_PATH)
        le       = joblib.load(ENCODER_PATH)
        features = joblib.load(FEATURES_PATH)
        return model, scaler, le, features, True
    except Exception as e:
        return None, None, None, None, False

model, scaler, le, features, model_loaded = load_model()

# ---- SIDEBAR ----
st.sidebar.image("https://img.icons8.com/fluency/96/shield.png", width=80)
st.sidebar.title("🛡️ SmartGuard AI")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigation", [
    "🏠 Dashboard",
    "📊 CSV Analysis",
    "🚨 Detection History",
    "📈 Model Performance",
    "ℹ️ About"
])

st.sidebar.markdown("---")
if model_loaded:
    st.sidebar.success("✅ Model Loaded")
else:
    st.sidebar.error("❌ Model Not Found")

# ============================================
# PAGE 1 — DASHBOARD
# ============================================
if page == "🏠 Dashboard":
    st.title("🛡️ SmartGuard AI — Network Intrusion Detection")
    st.markdown("**Real-Time Network Traffic Analysis using Machine Learning**")
    st.markdown("---")

    # Load detection logs
    if os.path.exists(LOG_FILE):
        df_log = pd.read_csv(LOG_FILE)
    else:
        df_log = pd.DataFrame(columns=['timestamp','src_ip','dst_ip','prediction','confidence','alert'])

    total    = len(df_log)
    normal   = len(df_log[df_log['prediction'] == 'Normal Traffic']) if total > 0 else 0
    attacks  = total - normal
    attack_pct = round((attacks/total)*100, 1) if total > 0 else 0

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📡 Total Flows", total)
    with col2:
        st.metric("✅ Normal Traffic", normal)
    with col3:
        st.metric("🚨 Attacks Detected", attacks)
    with col4:
        st.metric("⚠️ Attack Rate", f"{attack_pct}%")

    st.markdown("---")

    if total > 0:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Traffic Distribution")
            fig = px.pie(
                values=[normal, attacks],
                names=['Normal Traffic', 'Attacks'],
                color_discrete_sequence=['#00ff88', '#ff4444'],
                hole=0.4
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("🎯 Attack Types")
            attack_counts = df_log['prediction'].value_counts()
            fig2 = px.bar(
                x=attack_counts.index,
                y=attack_counts.values,
                color=attack_counts.index,
                color_discrete_sequence=px.colors.qualitative.Set1
            )
            fig2.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                showlegend=False
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("🕐 Recent Detections")
        recent = df_log.tail(10).iloc[::-1]
        for _, row in recent.iterrows():
            if row['prediction'] != 'Normal Traffic':
                st.markdown(f"""
                <div class="alert-box">
                🚨 <b>ATTACK</b> | {row['timestamp']} | 
                {row['src_ip']} → {row['dst_ip']} | 
                <b>{row['prediction']}</b> ({row['confidence']}%)
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="normal-box">
                ✅ <b>Normal</b> | {row['timestamp']} | 
                {row['src_ip']} → {row['dst_ip']}
                </div>""", unsafe_allow_html=True)
    else:
        st.info("📂 No detection logs yet. Run the sensor and engine, or use CSV Analysis.")

# ============================================
# PAGE 2 — CSV ANALYSIS
# ============================================
elif page == "📊 CSV Analysis":
    st.title("📊 CSV Traffic Analysis")
    st.markdown("Upload a network traffic CSV file to analyze with ML model")
    st.markdown("---")

    if not model_loaded:
        st.error("❌ Model not loaded! Check model files.")
    else:
        uploaded = st.file_uploader(
            "Upload CSV File",
            type=['csv'],
            help="Upload CICIDS2017 format CSV or flows.csv from sensor"
        )

        if uploaded:
            df = pd.read_csv(uploaded)
            st.success(f"✅ File loaded: {len(df)} rows, {len(df.columns)} columns")

            st.subheader("📋 Data Preview")
            st.dataframe(df.head(10))

            if st.button("🔍 Analyze Traffic", type="primary"):
                with st.spinner("Analyzing..."):
                    results = []
                    missing = [f for f in features if f not in df.columns]

                    if missing:
                        st.warning(f"⚠️ Missing {len(missing)} features — filling with 0")
                        for f in missing:
                            df[f] = 0

                    for _, row in df.iterrows():
                        try:
                            X = row[features].values.reshape(1, -1)
                            X_scaled = scaler.transform(X)
                            pred = model.predict(X_scaled)[0]
                            proba = model.predict_proba(X_scaled)[0]
                            label = le.inverse_transform([pred])[0]
                            confidence = round(max(proba)*100, 2)
                            results.append({
                                'src_ip': row.get('src_ip', 'N/A'),
                                'dst_ip': row.get('dst_ip', 'N/A'),
                                'prediction': label,
                                'confidence': confidence,
                                'alert': '🚨 ATTACK' if label != 'Normal Traffic' else '✅ Normal'
                            })
                        except:
                            pass

                    df_results = pd.DataFrame(results)
                    st.markdown("---")
                    st.subheader("🎯 Analysis Results")

                    total = len(df_results)
                    attacks = len(df_results[df_results['prediction'] != 'Normal Traffic'])
                    normal = total - attacks

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Flows", total)
                    col2.metric("✅ Normal", normal)
                    col3.metric("🚨 Attacks", attacks)

                    st.subheader("📊 Attack Distribution")
                    fig = px.pie(
                        df_results,
                        names='prediction',
                        color_discrete_sequence=px.colors.qualitative.Set1,
                        hole=0.4
                    )
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='white'
                    )
                    st.plotly_chart(fig)

                    st.subheader("📋 Detailed Results")
                    st.dataframe(df_results)

                    csv = df_results.to_csv(index=False)
                    st.download_button(
                        "⬇️ Download Results",
                        csv,
                        "smartguard_results.csv",
                        "text/csv"
                    )

# ============================================
# PAGE 3 — DETECTION HISTORY
# ============================================
elif page == "🚨 Detection History":
    st.title("🚨 Detection History")
    st.markdown("---")

    if os.path.exists(LOG_FILE):
        df_log = pd.read_csv(LOG_FILE)

        st.metric("Total Detections", len(df_log))

        attack_filter = st.selectbox(
            "Filter by Type",
            ["All"] + list(df_log['prediction'].unique())
        )

        if attack_filter != "All":
            df_log = df_log[df_log['prediction'] == attack_filter]

        st.dataframe(df_log.iloc[::-1], use_container_width=True)

        csv = df_log.to_csv(index=False)
        st.download_button("⬇️ Download History", csv, "detection_history.csv", "text/csv")
    else:
        st.info("No detection history yet.")

# ============================================
# PAGE 4 — MODEL PERFORMANCE
# ============================================
elif page == "📈 Model Performance":
    st.title("📈 Model Performance")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🏆 Random Forest")
        st.metric("Accuracy", "99.74%")
        st.metric("Precision", "100%")
        st.metric("Recall", "100%")
        st.metric("F1-Score", "100%")

    with col2:
        st.subheader("🌳 Decision Tree")
        st.metric("Accuracy", "99.65%")
        st.metric("Precision", "100%")
        st.metric("Recall", "99%")
        st.metric("F1-Score", "99%")

    st.markdown("---")
    st.subheader("📊 Algorithm Comparison")
    fig = go.Figure(data=[
        go.Bar(name='Random Forest', x=['Accuracy','Precision','Recall','F1'],
               y=[99.74, 100, 100, 100], marker_color='#00ff88'),
        go.Bar(name='Decision Tree', x=['Accuracy','Precision','Recall','F1'],
               y=[99.65, 100, 99, 99], marker_color='#4488ff')
    ])
    fig.update_layout(
        barmode='group',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        yaxis=dict(range=[98, 101])
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("🎯 Attack Classes")
    classes = ['Bots', 'Brute Force', 'DDoS', 'DoS',
               'Normal Traffic', 'Port Scanning', 'Web Attacks']
    st.write(f"**Total Classes:** {len(classes)}")
    for c in classes:
        st.write(f"• {c}")

# ============================================
# PAGE 5 — ABOUT
# ============================================
elif page == "ℹ️ About":
    st.title("ℹ️ About SmartGuard AI")
    st.markdown("---")

    st.markdown("""
    ## 🛡️ Project Overview
    **SmartGuard AI** is a Real-Time Network Intrusion Detection System
    that uses Machine Learning to identify malicious network traffic.

    ## 🎯 Objectives
    - Detect network attacks in real-time
    - Classify traffic into 7 categories
    - Provide visual analytics dashboard
    - Support both live monitoring and CSV analysis

    ## 🔬 Dataset
    - **CICIDS2017** — Canadian Institute for Cybersecurity
    - 2.5 Million network flow records
    - 52 network traffic features
    - 7 attack categories

    ## 🤖 Machine Learning
    - **Algorithm:** Random Forest Classifier
    - **Accuracy:** 99.74%
    - **Training samples:** 42,592
    - **Test samples:** 10,649

    ## 🏗️ System Architecture
    ```
    Kali Linux (Attacker)
         ↓
    Ubuntu Linux (Target + Sensor)
         ↓
    Feature Extraction (Scapy)
         ↓
    ML Detection Engine (Random Forest)
         ↓
    SmartGuard AI Dashboard (Streamlit)
    ```

    ## 👥 Team
    BS Cyber Security — MUET Jamshoro
    """)