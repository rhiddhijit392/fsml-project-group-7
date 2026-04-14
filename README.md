# 📈 Stock Price Prediction — FSML Group 7

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-Framework-green?logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Container-blue?logo=docker)
![ML](https://img.shields.io/badge/Machine%20Learning-Model-orange)
![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen)

A Full Stack Machine Learning project that predicts the **next day's closing stock price** using historical market data and deploys the model via a **FastAPI + Docker-based inference service**.

---

## 🚀 Problem Statement

Given historical daily stock data (OHLCV), predict the **next trading day's closing price**.

The system:
- Processes raw stock data
- Performs feature engineering
- Trains multiple ML models
- Selects the best model automatically
- Serves predictions through an API

---

## 🧠 Models Used

| Model | Description |
|------|------------|
| Random Forest | Ensemble of decision trees, robust and interpretable |
| XGBoost | Gradient boosting model, strong performance on tabular data |

The best model is selected based on evaluation metrics and saved for inference.

---

## 📂 Project Structure

```text
project/
├── app/              # FastAPI deployment layer
├── data/             # Dataset info and download link
├── logs/             # Runtime logs
├── models/           # Saved model (generated locally)
├── notebooks/        # EDA and experimentation
├── outputs/          # Evaluation plots and predictions
├── pipeline/         # End-to-end pipeline script
├── src/              # Core ML modules
├── config.yaml
├── Dockerfile
├── requirements.txt
└── README.md

```
> ⚠️ Note: The model file is mounted into the container using Docker volumes and is not included in the repository due to GitHub file size limits.


---

## ⚙️ Setup

```bash
git clone https://github.com/rhiddhijit392/fsml-project-group-7.git
cd fsml-project-group-7
pip install -r requirements.txt
```


📊 Dataset

The dataset is not included in this repository.

Download it from: 
data/dataset_link.txt

Place the file here: 
data/stock_details_5_years.csv

🔄 Run the ML Pipeline
python pipeline/pipeline.py

This performs:

Data loading
Preprocessing
Feature engineering
Model training
Model selection

👉 Output:

models/model_v1.pkl

🤖 Model File (Important)

The trained model is not stored in GitHub due to file size limitations.

To generate the model locally:

python pipeline/pipeline.py

🐳 Run with Docker
Step 1 — Build Docker image
docker build -t stock-predictor .
Step 2 — Run container (with model mount)
docker run -p 8000:8000 -v D:/your-path/fsml-project-group-7/models:/app/models stock-predictor

⚠️ Replace D:/your-path/... with your actual local path.

Step 3 — Access API

Open in browser:

http://127.0.0.1:8000/docs
🌐 API Usage
Endpoint
POST /predict
Sample Input
{
  "Open": 150.0,
  "High": 155.0,
  "Low": 148.0,
  "Close": 152.0,
  "Volume": 1200000,
  "Daily_Return": 0.01,
  "Price_Range": 7.0,
  "MA_5": 149.5,
  "Volume_Change": 0.05
}
Sample Output
{
  "predicted_next_close": 153.42,
  "current_close_approx": 150.0,
  "direction": "UP",
  "expected_change": 3.42,
  "model_used": "RandomForestRegressor"
}

📊 Evaluation Metrics

The models are evaluated using:

MAE (Mean Absolute Error)
RMSE (Root Mean Squared Error)
R² Score

Evaluation plots and results are stored in:

outputs/

🧠 Key Features
End-to-end ML pipeline
Time-series aware data splitting
Feature engineering (returns, moving averages, etc.)
Automatic model comparison and selection
FastAPI-based REST API
Dockerized deployment
External model handling (due to size constraints)

🏗️ System Architecture
🔹 Overview
┌──────────────────────┐
│   Stock Dataset      │
│ (Yahoo Finance CSV)  │
└─────────┬────────────┘
          ↓
┌──────────────────────┐
│   Data Pipeline      │
│ Load → Clean → FE    │
└─────────┬────────────┘
          ↓
┌──────────────────────┐
│   Model Training     │
│ RF / XGBoost         │
└─────────┬────────────┘
          ↓
┌──────────────────────┐
│   Best Model Saved   │
│ models/model_v1.pkl  │
└─────────┬────────────┘
          ↓
┌──────────────────────┐
│   FastAPI Service    │
│  /predict endpoint   │
└─────────┬────────────┘
          ↓
┌──────────────────────┐
│   Docker Container   │
│ (Inference Engine)   │
└─────────┬────────────┘
          ↓
┌──────────────────────┐
│ Client (Browser/API) │
│  http://localhost    │
└──────────────────────┘

![Architecture Diagram](https://raw.githubusercontent.com/ashishpatel26/fastapi-ml-deployment/main/architecture.png)

🔹 Flow Explanation
Data is collected and processed using a custom pipeline
Features like moving averages and returns are engineered
Multiple models are trained and evaluated
Best model is saved and used for inference
FastAPI exposes prediction endpoint
Docker ensures consistent deployment

💡 Key Design Decision

Due to GitHub file size limits, the trained model is:

Generated locally
Mounted into the Docker container at runtime

This ensures:

Lightweight repository
Scalable deployment
Clean version control

👨‍💻 Tech Stack
Python
Scikit-learn
XGBoost
FastAPI
Docker
Pandas, NumPy

📌 How to Run (Quick Summary)
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train model
python pipeline/pipeline.py

# 3. Build Docker image
docker build -t stock-predictor .

# 4. Run container
docker run -p 8000:8000 -v D:/path/models:/app/models stock-predictor

# 5. Open API
http://127.0.0.1:8000/docs
🚀 Future Improvements
Model optimization (LightGBM, smaller footprint)
Cloud deployment (AWS/GCP)
CI/CD pipeline integration
Real-time stock data integration

📄 License

This project is for educational and research purposes.