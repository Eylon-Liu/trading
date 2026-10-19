# Quant Research Terminal — container image.
#
# Built for a free-tier container host (Hugging Face Spaces, Render, Fly.io).
# The app is a long-lived WSGI server with a local SQLite store and on-disk
# caches, so it needs a real container; it does not fit a serverless runtime.
#
#   docker build -t quant-terminal .
#   docker run -p 7860:7860 -e DEMO_MODE=1 quant-terminal

FROM python:3.12-slim

# No apt layer. Every dependency ships a self-contained manylinux wheel —
# lxml bundles its own libxml2 and libxslt — and the seed download uses
# Python's urllib rather than curl, so there is nothing left to install. If a
# wheel ever stops being published the pip step should fail loudly rather than
# quietly compiling from source for ten minutes.

# Hugging Face Spaces runs containers as uid 1000. Creating that user here
# rather than running as root means the writable directories below are owned
# by the account that actually writes to them.
RUN useradd -m -u 1000 app

WORKDIR /home/app/trading

# Dependencies first: this layer is the slow one and changes far less often
# than the source, so keeping it above the COPY keeps rebuilds cheap.
COPY --chown=app:app requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir gunicorn

COPY --chown=app:app . .

# config.py creates db/, cache/ and reports/out/ at import time, so they must
# exist and be writable before the first import rather than at first write.
RUN mkdir -p db cache reports/out logs && chown -R app:app /home/app/trading

USER app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860 \
    DEMO_MODE=1

EXPOSE 7860

ENTRYPOINT ["./deploy/entrypoint.sh"]
