FROM python:3.11-slim

# Требуемые пакеты для pandas / sklearn
RUN apt-get update && apt-get install -y build-essential

WORKDIR /app

# Зависимости
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Сам сервис
COPY app.py .

EXPOSE 8000

CMD uvicorn app:app --host 0.0.0.0 --port 8000
