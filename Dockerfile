FROM python:3.11-slim

WORKDIR /app

# Install dependencies needed for some python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port
EXPOSE 7000

# Start server on 0.0.0.0 so it is accessible outside the container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7000"]
