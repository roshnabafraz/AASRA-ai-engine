import os
from datetime import datetime, timedelta, timezone
from io import BytesIO

from fastapi import FastAPI, HTTPException
import firebase_admin
from firebase_admin import credentials, firestore
from transformers import pipeline
import requests
from sklearn.ensemble import IsolationForest
import numpy as np

# ==========================================
# 0. FastAPI & Firebase Setup
# ==========================================
app = FastAPI(title="AASRA AI Engine", description="Disaster Management AI Backend")

# Initialize Firebase Admin using the local service account key
# Ensure 'firebase-key.json' is located in the same directory as this file
firebase_err = "Not executed"
try:
    path_tried = "None"
    
    # Check if running locally (could be firebase-key.json or .json.json)
    if os.path.exists("firebase-key.json"):
        path_tried = "./firebase-key.json"
        cred = credentials.Certificate("firebase-key.json")
    elif os.path.exists("firebase-key.json.json"):
        path_tried = "./firebase-key.json.json"
        cred = credentials.Certificate("firebase-key.json.json")
        
    # Check if running on Render Cloud Native or Docker
    elif os.path.exists("/etc/secrets/firebase-key.json"):
        path_tried = "/etc/secrets/firebase-key.json"
        cred = credentials.Certificate("/etc/secrets/firebase-key.json")
    elif os.path.exists("/etc/secrets/firebase-key.json.json"):
        path_tried = "/etc/secrets/firebase-key.json.json"
        cred = credentials.Certificate("/etc/secrets/firebase-key.json.json")
        
    else:
        # Fallback to check if it's injected to /app/ by some Docker/Render quirk
        files_in_etc = os.listdir("/etc/secrets") if os.path.exists("/etc/secrets") else "no_etc_secrets"
        files_in_root = os.listdir(".")
        raise RuntimeError(f"Key not found! /etc/secrets: {files_in_etc}. Root: {files_in_root}")

    # Prevent re-initialization error if running in a hot-reloading environment
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
    db = firestore.client()
except Exception as e:
    print(f"Warning: Firebase not initialized. {e}")
    # Store exact error and the path we tried to load so we can debug cleanly
    firebase_err = f"[{type(e).__name__}] {str(e)} | Path: {path_tried}"
    db = None

# ==========================================
# 1. Scikit-Learn Setup (Advanced Fraud Detection)
# ==========================================
print("Training Scikit-Learn IsolationForest model (Advanced)...")
# Create a dummy training array representing multi-dimensional features:
# [total_reports_last_5_mins, average_text_length, time_variance_seconds]
# Normal users send 1-4 reports with reasonable text length and high variance between reports.
# Spammers send 10+ reports with very short text (e.g. "help") and very low variance.
np.random.seed(42)
X_normal = np.column_stack((
    np.random.poisson(lam=1.5, size=80) + 1,        # 1-4 reports
    np.random.normal(loc=120, scale=30, size=80),   # 90-150 chars
    np.random.normal(loc=300, scale=50, size=80)    # high time variance
))
X_fraud = np.column_stack((
    np.random.poisson(lam=15, size=20) + 10,        # 15-30 reports
    np.random.normal(loc=15, scale=5, size=20),     # 10-20 chars (bot-like)
    np.random.normal(loc=5, scale=2, size=20)       # 3-7 seconds variance (scripted)
))
X_train = np.vstack((X_normal, X_fraud))

fraud_model = IsolationForest(contamination=0.20, random_state=42)
fraud_model.fit(X_train)
print("Scikit-Learn model trained successfully.")

# ==========================================
# 2. Transformers Setup (Advanced NLP Priority)
# ==========================================
print("Loading HuggingFace Zero-Shot Classification Model...")
# We use a lightweight cross-encoder for zero-shot classification to understand urgency deeply
priority_classifier = pipeline("zero-shot-classification", model="cross-encoder/nli-distilroberta-base")
print("HuggingFace model loaded successfully.")

# ==========================================
# 3. AI Functions
# ==========================================

def analyze_priority(text: str) -> str:
    """
    1. Uses a Zero-Shot Classification Pipeline to assign 'High', 'Medium', or 'Low' 
       based on the actual semantic urgency of the text.
    """
    if not text or len(text.strip()) < 3:
        return 'Low'
        
    try:
        candidate_labels = ["critical life-threatening emergency", "moderate incident requiring assistance", "low priority information"]
        result = priority_classifier(text, candidate_labels)
        top_label = result['labels'][0]
        
        if top_label == "critical life-threatening emergency":
            return 'High'
        elif top_label == "moderate incident requiring assistance":
            return 'Medium'
        else:
            return 'Low'
    except Exception as e:
        print(f"Error in zero-shot classification: {e}")
        # Fallback to basic length heuristic if model fails
        return 'Medium' if len(text) > 20 else 'Low'

def detect_fraud(user_id: str) -> bool:
    """
    2. Queries Firestore for the user's recent reports. Extracts features 
       (count, avg length, variance) and predicts using Scikit-Learn IsolationForest.
    """
    if not user_id or db is None:
        return False
        
    try:
        # Define the time window: 5 minutes ago from now (UTC time)
        five_mins_ago = datetime.now(timezone.utc) - timedelta(minutes=5)
        
        reports_ref = db.collection('reports')
        
        # Using stream() to retrieve the documents
        query = reports_ref.where('user_id', '==', user_id).where('timestamp', '>=', five_mins_ago)
        recent_reports = list(query.stream())
        report_count = len(recent_reports)
        
        if report_count == 0:
            return False
            
        # Calculate average text length
        lengths = [len(r.to_dict().get('description', '')) for r in recent_reports]
        avg_length = np.mean(lengths) if lengths else 0
        
        # Calculate time variance
        if report_count > 1:
            timestamps = sorted([r.to_dict().get('timestamp').timestamp() for r in recent_reports if r.to_dict().get('timestamp')])
            if len(timestamps) > 1:
                diffs = np.diff(timestamps)
                variance = np.var(diffs)
            else:
                variance = 300
        else:
            variance = 300
            
        # Predict outliers using the loaded IsolationForest model (3D Expected)
        features = np.array([[report_count, avg_length, variance]])
        prediction = fraud_model.predict(features)
        
        # IsolationForest returns 1 for normal and -1 for outliers
        is_fraud = bool(prediction[0] == -1)
        return is_fraud
        
    except Exception as e:
        print(f"Error detecting fraud for user {user_id}: {e}")
        return False

# ==========================================
# 4. The Endpoint
# ==========================================

@app.post("/process_report/{report_id}")
async def process_report(report_id: str):
    """
    Fetches the report from Firestore, runs all 3 AI functions, 
    and updates the Firestore document with the results (priority, AI_Verified, fraud_flag, status).
    """
    if db is None:
        raise HTTPException(status_code=500, detail=f"Firebase DB failed to initialize. Error details: {firebase_err}")
        
    try:
        # Fetch the report document from Firestore
        doc_ref = db.collection('reports').document(report_id)
        doc = doc_ref.get()
        
        if not doc.exists:
            raise HTTPException(status_code=404, detail=f"Report {report_id} not found in Firestore.")
            
        data = doc.to_dict()
        
        # Extract necessary fields (using default values if missing)
        description = data.get('description', '')
        image_url = data.get('image_url', '')
        user_id = data.get('user_id', '')
        
        print(f"Processing report: {report_id} for user: {user_id}")
        
        # Run AI Functions
        priority = analyze_priority(description)
        fraud_flag = detect_fraud(user_id) if user_id else False
        
        # Determine current status mapping based on the fraud detection outcome.
        status = 'Rejected' if fraud_flag else 'Processed'
        
        # Prepare the update dictionary
        updates = {
            'priority': priority,
            'fraud_flag': fraud_flag,
            'status': status
        }
        
        # Update the existing Firestore document with the AI results
        doc_ref.update(updates)
        
        print(f"Report {report_id} successfully updated with results: {updates}")
        
        return {
            "message": "Report processed successfully",
            "report_id": report_id,
            "results": updates
        }
        
    except HTTPException as he:
        # Re-raise known HTTP exceptions
        raise he
    except Exception as e:
        print(f"Server error processing report {report_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.get("/")
def health_check():
    return {"status": "AASRA AI Engine is running and ready for reports."}
