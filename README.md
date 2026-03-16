# AASRA AI Engine 🧠⚙️

![AASRA AI Engine Cover](placeholder_image_url_here)
*Caption: AASRA AI Engine Processing Block.*

The **AASRA AI Engine** is the intelligent backbone of the AASRA disaster management ecosystem. Built with **FastAPI** and powered by **PyTorch** and **Scikit-Learn**, it provides critical automated analysis for crisis reports submitted via the Android app. 

It seamlessly integrates with Firebase Firestore to process incoming reports, verifying images, detecting fraudulent submissions, and assigning priority levels, ensuring that genuine emergencies are escalated instantly.

---

## 🚀 Key Features

*   **Real-time Report Processing**: Processes incoming disaster reports automatically via REST endpoint.
*   **Priority Analysis (NLP)**: Uses intelligent keyword processing to categorize reports as `High`, `Medium`, or `Low` priority.
*   **Image Verification**: Integrates a pre-trained **MobileNetV2** model (via PyTorch) to verify the authenticity of uploaded disaster images.
*   **Fraud & Spam Detection**: Utilizes an **Isolation Forest** model from Scikit-Learn to identify abnormal reporting frequencies and flag potential spam or fraudulent user accounts.
*   **Cloud Integration**: Seamlessly syncs results back to **Firebase Firestore**.

---

## 🛠️ Tech Stack & Dependencies

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Firebase](https://img.shields.io/badge/Firebase-FFCA28?style=for-the-badge&logo=firebase&logoColor=black)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

*   **Framework**: FastAPI, Uvicorn
*   **Machine Learning**: PyTorch, Torchvision, Scikit-Learn
*   **Data Processing**: Numpy, Pillow
*   **Database Integration**: Firebase Admin SDK

---

## 📡 API Endpoints

### 1. Process Report Endpoint
**`POST /process_report/{report_id}`**

Fetches a specific report from Firestore, runs it through the AI functions, and updates the database record with the analysis results.

**Path Parameters:**
*   `report_id` (string): The unique Document ID of the report in Firestore.

**AI Processing Pipeline:**
1.  **Priority**: Identifies keywords to mark priority (e.g., "fire", "earthquake" = `High`).
2.  **AI_Verified**: Downloads image url and runs inference via MobileNetV2.
3.  **Fraud_Flag**: Queries Firestore for recent reports by `user_id` and detects anomalies using Isolation Forest.
4.  Updates document status to `Processed` (or `Rejected` if marked as fraud).

**Successful Response (200 OK):**
```json
{
  "message": "Report processed successfully",
  "report_id": "abc123xyz",
  "results": {
    "priority": "High",
    "AI_Verified": true,
    "fraud_flag": false,
    "status": "Processed"
  }
}
```

### 2. Health Check
**`GET /`**

Returns the operational status of the engine.

---

## 💻 Local Setup Instructions

Follow these step-by-step instructions to get the AI Engine running locally on your machine.

### Prerequisites
*   Python 3.8+ installed on your machine.
*   Your Firebase Service Account Key JSON file.

### Step 1: Clone the repository
Navigate to the AI engine directory:
```bash
cd aasra-ai-engine
```

### Step 2: Set up a Virtual Environment
It is recommended to run the engine in an isolated Python virtual environment.
```bash
python -m venv venv
```
Activate it:
*   **Windows**: `venv\Scripts\activate`
*   **macOS/Linux**: `source venv/bin/activate`

### Step 3: Install Dependencies
Install all required libraries using pip:
```bash
pip install -r requirements.txt
```

### Step 4: Configure Firebase
1.  Obtain your `firebase-admin-sdk` credentials JSON file from the Firebase Console (Project Settings > Service Accounts).
2.  Rename the file to `firebase-key.json` and place it in the root directory of the engine (`aasra-ai-engine/`).

### Step 5: Run the Server
Start the FastAPI server using Uvicorn:
```bash
uvicorn main:app --reload
```
The server will start locally at `http://localhost:8000`. You can access the auto-generated Swagger UI documentation at `http://localhost:8000/docs`.

---

## 📸 Application Images
*(Images to be added here)*

![AI Processing Graph](placeholder_image_url_here)
*Caption: Real-time fraud detection and image processing metrics.*

---
*Created for the AASRA Final Year Project.*
