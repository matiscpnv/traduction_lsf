FROM python:3.10-slim

# Install system dependencies for OpenCV and Mediapipe
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose for Render
ENV PORT=10000

CMD ["gunicorn", "--bind", "0.0.0.0:10000", "app:app"]
