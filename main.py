import os
from datetime import datetime, timedelta, timezone
from io import BytesIO

from fastapi import FastAPI, HTTPException
import firebase_admin
from firebase_admin import credentials, firestore
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
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
# 1. PyTorch Setup (Image Verification)
# ==========================================
print("Loading MobileNetV2 model...")
# Load pre-trained MobileNetV2 in evaluation mode
try:
    # Use newer Weights API if torchvision is updated, otherwise fallback to pretrained=True
    weights = models.MobileNet_V2_Weights.DEFAULT
    mobilenet_model = models.mobilenet_v2(weights=weights)
except AttributeError:
    mobilenet_model = models.mobilenet_v2(pretrained=True)
mobilenet_model.eval()

# Standard image transforms for MobileNetV2
image_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
print("PyTorch model loaded successfully.")

# ==========================================
# 2. Scikit-Learn Setup (Fraud Detection)
# ==========================================
print("Training Scikit-Learn IsolationForest model...")
# Create a dummy training array representing user reporting frequency.
# Normal users send 1-2 reports; spammers send 10+.
# The array contains individual report counts for various simulated users.
X_train = np.array([
    [1], [2], [1], [1], [0], [3], [2], [1], [2],  # Normal report counts
    [14], [25], [11], [30]                        # Abnormally high report counts (spam/fraud)
])

# Initialize and fit the Scikit-learn IsolationForest model
# 'contamination' is the expected proportion of outliers (fraudsters)
fraud_model = IsolationForest(contamination=0.15, random_state=42)
fraud_model.fit(X_train)
print("Scikit-Learn model trained successfully.")

# ==========================================
# 3. AI Functions
# ==========================================

def analyze_priority(text: str) -> str:
    """
    1. A rule-based NLP function returning 'High', 'Medium', or 'Low' 
       based on crisis keywords.
    """
    if not text:
        return 'Low'
        
    text_lower = text.lower()
    
    # Priority keyword lists
    high_keywords = ['fire', 'earthquake', 'flood', 'trapped', 'blood', 'dying', 'emergency', 'help', 'critical', 'blast', 'collapse']
    medium_keywords = ['injury', 'accident', 'damage', 'power', 'water', 'food', 'shelter', 'blocked', 'hurt']
    
    # Check for high priority keywords first
    if any(keyword in text_lower for keyword in high_keywords):
        return 'High'
    # Then check for medium priority keywords
    elif any(keyword in text_lower for keyword in medium_keywords):
        return 'Medium'
    else:
        return 'Low'

def verify_image(image_url: str) -> bool:
    """
    2. Downloads an image using requests, runs it through the PyTorch MobileNetV2 tensor, 
       and returns True if successful.
    """
    if not image_url:
        return False
        
    try:
        # Download the image via requests
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        
        # Load image into Pillow
        image = Image.open(BytesIO(response.content)).convert("RGB")
        
        # Apply standard transforms and prepare tensor batch (1, C, H, W)
        input_tensor = image_transforms(image).unsqueeze(0)
        
        # Pass the tensor through MobileNetV2
        with torch.no_grad():
            output = mobilenet_model(input_tensor)
            
        # If no exceptions were raised during tensor processing and model forward pass,
        # consider the image successfully verified by AI.
        return True
    except Exception as e:
        print(f"Error verifying image from URL {image_url}: {e}")
        return False

def detect_fraud(user_id: str) -> bool:
    """
    3. Queries Firestore for the user's report count in the last 5 minutes. 
       Passes this count into the Scikit-learn IsolationForest model to return True (fraud) or False.
    """
    if not user_id or db is None:
        return False
        
    try:
        # Define the time window: 5 minutes ago from now (UTC time)
        five_mins_ago = datetime.now(timezone.utc) - timedelta(minutes=5)
        
        # Query Firestore for reports generated by the user within the last 5 minutes
        # Note: A composite index on 'user_id' and 'timestamp' may be required in Firestore.
        reports_ref = db.collection('reports')
        
        # Using stream() to retrieve the documents
        query = reports_ref.where('user_id', '==', user_id).where('timestamp', '>=', five_mins_ago)
        recent_reports = list(query.stream())
        report_count = len(recent_reports)
        
        # Predict outliers using the loaded IsolationForest model
        # The model expects a 2D array, so we reshape the count into [[report_count]]
        prediction = fraud_model.predict(np.array([[report_count]]))
        
        # IsolationForest returns 1 for normal (inliers) and -1 for outliers (fraud/spam)
        is_fraud = bool(prediction[0] == -1)
        return is_fraud
        
    except Exception as e:
        print(f"Error detecting fraud for user {user_id}: {e}")
        # Default to false so we don't accidentally block legitimate users when an error occurs
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
        
        # Run AI Functions (Sections 3.1, 3.2, 3.3)
        priority = analyze_priority(description)
        ai_verified = verify_image(image_url) if image_url else False
        fraud_flag = detect_fraud(user_id) if user_id else False
        
        # Determine current status mapping based on the fraud detection outcome.
        status = 'Rejected' if fraud_flag else 'Processed'
        
        # Prepare the update dictionary with exactly the fields requested
        updates = {
            'priority': priority,
            'AI_Verified': ai_verified,
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
