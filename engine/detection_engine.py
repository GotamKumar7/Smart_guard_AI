# ============================================
# SMARTGUARD AI — PHASE 6
# DETECTION ENGINE
# ============================================

import pandas as pd
import numpy as np
import joblib
import os
import time
from datetime import datetime

# Paths
MODEL_PATH    = r'D:\smarguard ai\ml_model\smartguard_model.pkl'
SCALER_PATH   = r'D:\smarguard ai\ml_model\scaler.pkl'
ENCODER_PATH  = r'D:\smarguard ai\ml_model\label_encoder.pkl'
FEATURES_PATH = r'D:\smarguard ai\ml_model\feature_names.pkl'
FLOWS_CSV     = r'D:\smarguard ai\sensor\flows.csv'
LOG_FILE      = r'D:\smarguard ai\logs\detections.csv'

# Load model files
print("⏳ Loading SmartGuard AI Model...")
model    = joblib.load(MODEL_PATH)
scaler   = joblib.load(SCALER_PATH)
le       = joblib.load(ENCODER_PATH)
features = joblib.load(FEATURES_PATH)
print("✅ Model loaded successfully!")
print(f"🎯 Classes: {list(le.classes_)}")

# Create logs folder
os.makedirs(r'D:\smarguard ai\logs', exist_ok=True)

def predict_flows(df):
    results = []
    for _, row in df.iterrows():
        try:
            src_ip    = row.get('src_ip', 'Unknown')
            dst_ip    = row.get('dst_ip', 'Unknown')
            timestamp = row.get('timestamp', datetime.now())

            X = row[features].values.reshape(1, -1)
            X_scaled = scaler.transform(X)

            pred   = model.predict(X_scaled)[0]
            proba  = model.predict_proba(X_scaled)[0]
            label  = le.inverse_transform([pred])[0]
            confidence = round(max(proba) * 100, 2)

            result = {
                'timestamp'  : timestamp,
                'src_ip'     : src_ip,
                'dst_ip'     : dst_ip,
                'prediction' : label,
                'confidence' : confidence,
                'alert'      : 'ALERT' if label != 'Normal Traffic' else 'Normal'
            }
            results.append(result)

            alert = "🚨 ATTACK DETECTED" if label != 'Normal Traffic' else "✅ Normal"
            print(f"{alert} | {src_ip} → {dst_ip} | {label} ({confidence}%)")

        except Exception as e:
            print(f"⚠️ Error: {e}")

    return results

def save_results(results):
    if results:
        df = pd.DataFrame(results)
        write_header = not os.path.exists(LOG_FILE)
        df.to_csv(LOG_FILE, mode='a', header=write_header, index=False)
        print(f"💾 {len(results)} detections saved!")

def run_engine():
    print("\n" + "="*55)
    print("🚀 SmartGuard AI Detection Engine Running")
    print("="*55)
    print(f"📂 Watching: {FLOWS_CSV}")
    print("⏳ Waiting for flows... (Ctrl+C to stop)\n")

    processed = set()

    while True:
        try:
            if os.path.exists(FLOWS_CSV):
                df = pd.read_csv(FLOWS_CSV)
                new_rows = df[~df.index.isin(processed)]

                if len(new_rows) > 0:
                    print(f"\n📊 Processing {len(new_rows)} new flows...")
                    results = predict_flows(new_rows)
                    save_results(results)
                    processed.update(new_rows.index)

            time.sleep(5)

        except KeyboardInterrupt:
            print("\n⛔ Detection Engine Stopped.")
            break
        except Exception as e:
            print(f"⚠️ Engine error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    run_engine()