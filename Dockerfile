FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "src/train.py"]

Create requirements.txt:

xgboost
pandas
numpy
scikit-learn
joblib

Create .dockerignore:

.venv/
__pycache__/
*.pyc
.git/
models/
data/