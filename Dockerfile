# REST API for the AI Economic Analyst agents.
#   docker build -t economic-analyst-api .
#   docker run -p 8000:8000 -e CEREBRAS_API_KEY=... economic-analyst-api
# Then open http://localhost:8000/docs
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first so code changes don't reinstall them
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY *.py economic_data.csv economy.db ./
COPY reports/ reports/

# Build the report index at image build time. This also downloads
# Chroma's local embedding model into the image, so the container
# needs no extra downloads when it starts.
RUN useradd --create-home appuser \
    && chown -R appuser /app
USER appuser
RUN python build_index.py

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
