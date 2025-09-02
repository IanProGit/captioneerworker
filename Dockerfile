FROM python:3.11-slim

# System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg curl && \
    rm -rf /var/lib/apt/lists/*

# App
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app.py /app/app.py

# Runtime
ENV PYTHONUNBUFFERED=1 PORT=8080
EXPOSE 8080
CMD ["python", "app.py"]