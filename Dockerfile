#select a python version
FROM python:3.12-slim

#install OpenMP for xgboost
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# ✅ RUN FASTAPI (NOT TRAINING)
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]