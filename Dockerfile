FROM registry.hub.docker.com/library/python:3.12.13-slim

# Create a directory for the application
WORKDIR /home/vcap/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file initially to leverage Docker cache
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . . 

CMD ["fastapi", "run", "main.py"]
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
