FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m nltk.downloader wordnet

COPY . .

# Ensure scripts are executable
RUN chmod +x cmd/*.sh

CMD ["/bin/bash"]
