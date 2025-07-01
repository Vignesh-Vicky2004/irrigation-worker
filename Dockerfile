# Use slim Python image
FROM python:3.10-slim

# Create app directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy main app and model
COPY main.py .
COPY tamil_nadu_irrigation_model.pkl .

# Expose port (for FastAPI)
EXPOSE 8000

# Run FastAPI with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
