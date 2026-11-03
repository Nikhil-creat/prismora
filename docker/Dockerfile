FROM python:3.11-slim AS base
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY backend/requirements.txt .
# CPU torch by default; swap the base image for nvidia/cuda to use a GPU
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt
FROM base AS runtime
COPY backend/ .
RUN useradd -m app && chown -R app /app
USER app
EXPOSE 8000
CMD ["uvicorn","main:app","--host","0.0.0.0","--port","8000"]
