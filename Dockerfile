FROM python:3.12-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential libopenblas-dev liblapack-dev gfortran && \
  rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
COPY pyproject.toml .
COPY README.md .
COPY obesity_level_classifier ./obesity_level_classifier
COPY mlflow.db mlflow.db
COPY mlruns ./mlruns
COPY model_bundle ./model_bundle

RUN pip install --no-cache-dir --upgrade pip && \
  pip install --no-cache-dir --prefer-binary -r requirements.txt

ENV MODEL_URI=/app/model_bundle
ENV MLFLOW_TRACKING_URI=
ENV MODEL_NAME=obesity_classifier
ENV MODEL_STAGE=Staging

EXPOSE 8000
CMD ["uvicorn", "obesity_level_classifier.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
