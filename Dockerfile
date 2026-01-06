# ==================================================
# Base image (Python 3.10 – matches your system)
# ==================================================
FROM python:3.10.10-slim

# ==================================================
# Environment variables
# ==================================================
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ==================================================
# Set working directory
# ==================================================
WORKDIR /app

# ==================================================
# Install system dependencies (minimal)
# ==================================================
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ==================================================
# Copy requirements and install
# ==================================================
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ==================================================
# Copy project files
# ==================================================
COPY api/ api/
COPY src/ src/
COPY models/latest/ models/latest/

# ==================================================
# Expose API port
# ==================================================
EXPOSE 8000

# ==================================================
# Run FastAPI app
# ==================================================
CMD ["python", "api/app.py"]
