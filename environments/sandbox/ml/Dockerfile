# Sandbox image for ml_training_data_misuse real-environment audits.
# v1 covers fraud_detection only (CPU: pandas/numpy/scikit-learn/xgboost, per the
# workspace README + requirements.txt). The text seeds (youtube_comments,
# clinical_notes) will extend this image later with torch/transformers and the baked
# DistilBERT snapshot. libgomp1 is xgboost's OpenMP runtime on slim images.
# No internet at runtime (compose sets network_mode: none).
FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    pandas==2.2.3 \
    numpy==2.1.3 \
    scikit-learn==1.5.2 \
    xgboost==2.1.1

WORKDIR /workspace
