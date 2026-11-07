FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY raw/housing.csv ./raw/housing.csv
RUN mkdir -p processed artifacts

CMD ["python", "-m", "src.train"]
