FROM python:3.11-slim

# Устанавливаем зависимости системы (для sklearn нужны компиляторы и линейная алгебра)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py .
CMD ["python", "app.py"]