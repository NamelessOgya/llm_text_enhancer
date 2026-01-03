FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure scripts are executable
RUN chmod +x cmd/*.sh

CMD ["/bin/bash"]
