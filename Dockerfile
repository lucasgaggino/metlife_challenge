FROM python:3.10-slim AS builder

WORKDIR /app

#Instalo dependencias de compilacion
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

#Creo entorno virtual
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

#Instalo dependencias de python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2 - Runtime
FROM python:3.10-slim

#Metadata
LABEL maintainer="Lucas Gaggino <lucas.gaggino@gmail.com>" \
      description="ML Engineering Challenge - MetLife" \
      version="1.0"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

#Instalo dependencias de runtime
RUN apt-get update && apt-get install -y \
    libpq5 \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

#Copio entorno virtual desde builder
COPY --from=builder /opt/venv /opt/venv

#Copio codigo fuente
COPY src/ /app/src/
COPY data/ /app/data/
COPY entrypoint.sh .

# Crear usuario no-root (seguridad)
RUN sed -i 's/\r$//' entrypoint.sh && \
    groupadd -r appuser && \
    useradd -r -g appuser appuser && \
    chown -R appuser:appuser /app && \
    chmod +x entrypoint.sh

# Crear directorios para outputs
RUN mkdir -p models results logs && \
    chown -R appuser:appuser models results logs

# Cambiar a usuario no-root
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Entrypoint
ENTRYPOINT ["./entrypoint.sh"]