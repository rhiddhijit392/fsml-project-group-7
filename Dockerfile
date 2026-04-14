#select a python vesrion
FROM python:3.12-slim

#install OpenMP for xgboost & clear apt cached packages to image reduce size
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

#set working directory inside container
WORKDIR /app

# Disable stdout buffering so logs are flushed instantly
ENV PYTHONUNBUFFERED=1

#install necessary python libraries
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
    
#copy rest of the project files
COPY . .

#run pipeline
CMD ["python", "pipeline/pipeline.py"]
