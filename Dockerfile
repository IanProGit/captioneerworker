FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py ./

# Render sets $PORT; bind to it with Gunicorn
CMD ["gunicorn","-w","2","-k","gthread","-b","0.0.0.0:$PORT","app:app"]
