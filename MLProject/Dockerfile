# Gunakan base image Python versi ringan
FROM python:3.12-slim

# Set direktori kerja di dalam container
WORKDIR /app

# Salin semua file ke dalam container
COPY . .

# Install dependensi sistem yang mungkin dibutuhkan ML/Sklearn
RUN apt-get update && \
    apt-get install -y build-essential && \
    apt-get clean

# Install dependensi Python
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Default command untuk menjalankan training
CMD ["python", "modelling.py", "--data_path", "personality.csv", "--n_estimators", "100", "--max_depth", "10"]
