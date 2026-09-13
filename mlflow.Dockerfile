FROM python:3.12-slim-bookworm

WORKDIR /mlflow

RUN pip install --no-cache-dir mlflow==3.14.0

RUN mkdir -p /mlflow/mlartifacts

EXPOSE 5000

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--workers", "1", "--backend-store-uri", "sqlite:////mlflow/mlflow.db", "--serve-artifacts", "--artifacts-destination", "/mlflow/mlartifacts", "--allowed-hosts", "localhost:5000,127.0.0.1:5000,mlflow-server:5000"]