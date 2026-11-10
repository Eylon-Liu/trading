# Quant Research Terminal — web app + scheduler image.
#   docker compose up -d        (see docker-compose.yml and docs/remote-access.md)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
# Data lives in volumes (see docker-compose.yml); the image holds code only.
RUN useradd --create-home --uid 1000 quant \
 && mkdir -p db cache reports/out logs \
 && chown -R quant:quant /app
USER quant

EXPOSE 8050
HEALTHCHECK --interval=60s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8050/healthz', timeout=4)"
CMD ["python", "app.py"]
