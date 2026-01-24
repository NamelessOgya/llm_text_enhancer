FROM python:3.11-slim

# Install system dependencies for fonts
RUN apt-get update && apt-get install -y \
    fonts-dejavu \
    libfreetype6-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m nltk.downloader wordnet

COPY . .

# Ensure scripts are executable
RUN chmod +x cmd/*.sh

CMD ["/bin/bash"]
