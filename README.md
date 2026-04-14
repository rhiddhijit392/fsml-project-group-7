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
|-------|-------------|
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

> ⚠️ The model file is mounted into the container using Docker volumes and is not included in the repository due to GitHub file size limits.

---

## ⚙️ Setup

```bash
git clone https://github.com/rhiddhijit392/fsml-project-group-7.git
cd fsml-project-group-7
pip install -r requirements.txt
```

---

## 📊 Dataset

The dataset is not included in this repository.

- Download link: `data/dataset_link.txt`
- Place the file at: `data/stock_details_5_years.csv`

---

## 🔄 Run the ML Pipeline

```bash
python pipeline/pipeline.py
```

This performs: data loading → preprocessing → feature engineering → model training → model selection.

**Output:** `models/model_v1.pkl`

---

## 🐳 Run with Docker

**Step 1 — Build the image**

```bash
docker build -t stock-predictor .
```

**Step 2 — Run the container**

```bash
docker run -p 8000:8000 -v /your/local/path/models:/app/models stock-predictor
```

> ⚠️ Replace `/your/local/path/` with your actual path to the project folder.

**Step 3 — Access the API**

Open in your browser: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🌐 API Usage

**Endpoint:** `POST /predict`

**Sample Input:**

```json
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
```

**Sample Output:**

```json
{
  "predicted_next_close": 153.42,
  "current_close_approx": 150.0,
  "direction": "UP",
  "expected_change": 3.42,
  "model_used": "RandomForestRegressor"
}
```

---

## 📊 Evaluation Metrics

Models are evaluated using:

- **MAE** — Mean Absolute Error
- **RMSE** — Root Mean Squared Error
- **R² Score**

Plots and results are saved to `outputs/`.

---

## 🏗️ System Architecture

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

---

## 💡 Key Design Decision

Due to GitHub file size limits, the trained model is generated locally and mounted into the Docker container at runtime. This ensures a lightweight repository, clean version control, and flexible deployment.

---

## 👨‍💻 Tech Stack

- Python 3.12
- Scikit-learn
- XGBoost
- FastAPI + Uvicorn
- Docker
- Pandas, NumPy

---

## 📌 Quick Start Summary

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model
python pipeline/pipeline.py

# 3. Build Docker image
docker build -t stock-predictor .

# 4. Run container (update path)
docker run -p 8000:8000 -v /your/local/path/models:/app/models stock-predictor

# 5. Open API docs
# http://127.0.0.1:8000/docs
```

---

## 🚀 Future Improvements

- Model optimization (LightGBM, smaller footprint)
- Cloud deployment (AWS / GCP)
- Real-time stock data integration
- Enhanced CI/CD pipeline

---

## 📄 License

This project is for educational and research purposes.