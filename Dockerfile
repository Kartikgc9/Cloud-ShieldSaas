# Use official slim Python runtime as a parent image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        postgresql-client \
        build-essential \
        libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements_saas.txt ./
RUN pip install --no-cache-dir -r requirements_saas.txt

# Copy application source code
COPY . ./

# Create necessary directories
RUN mkdir -p models data logs

# Ensure wsgi.py is executable
RUN chmod +x wsgi.py

# Expose application port
EXPOSE 5000

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/ || exit 1

# Launch the application with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "120", "wsgi:app"]
