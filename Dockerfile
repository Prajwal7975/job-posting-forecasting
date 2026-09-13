FROM python:3.12-slim-bookworm

# ==========================================================
# Environment
# ==========================================================

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ==========================================================
# Working directory
# ==========================================================

WORKDIR /app

# ==========================================================
# System dependencies + security updates
# ==========================================================

RUN apt-get update \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/*

# ==========================================================
# Python dependencies
# ==========================================================

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ==========================================================
# Application source
# ==========================================================

COPY src ./src
COPY api ./api

# ==========================================================
# Runtime user + writable directories
# ==========================================================

RUN useradd \
    --create-home \
    --shell /usr/sbin/nologin \
    appuser \
    && mkdir -p /app/logs \
    && chown -R appuser:appuser /app

USER appuser

# ==========================================================
# Port
# ==========================================================

EXPOSE 8000

# ==========================================================
# FastAPI
# ==========================================================

CMD ["uvicorn", "api.salary_api:app", "--host", "0.0.0.0", "--port", "8000"]