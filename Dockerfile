# conso-forecast: API + demo page, with the model trained while the image is built.
# Build: docker compose build     Run: docker compose up     Then open http://127.0.0.1:8000/
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# 1. Dependencies first: this layer is cached until requirements-api.txt changes.
#    torch comes from the CPU-only index (the default Linux wheel ships GPU libraries: several GB).
COPY requirements-api.txt .
RUN grep '^torch==' requirements-api.txt > torch-requirement.txt \
 && pip install --index-url https://download.pytorch.org/whl/cpu -r torch-requirement.txt \
 && grep -v '^torch==' requirements-api.txt > other-requirements.txt \
 && pip install -r other-requirements.txt \
 && rm torch-requirement.txt other-requirements.txt

# Never run as root inside the container. The user is created BEFORE the files, which are copied with their
# final owner: a `chown -R` after the fact would duplicate every file in a new layer (+34 MB measured).
RUN useradd --create-home --uid 1000 app && chown app:app /app
USER app

# 2. Data: RTE eCO2mix, open licence (Etalab). Cached until the download code changes.
COPY --chown=app:app src/__init__.py src/config.py src/data.py src/
RUN python -m src.data

# 3. Model: trained here, on the data downloaded above (writes models/gru_final.pt). ~2 minutes.
COPY --chown=app:app src/ src/
RUN python -m src.train_final

# 4. API and page (copied last: editing the page does not retrain the model).
COPY --chown=app:app api/ api/

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status == 200 else 1)"]

# 0.0.0.0 = all interfaces: with the default 127.0.0.1 the server would be unreachable from outside the container.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
