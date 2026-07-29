FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    KEYVISION_CONFIG=configs/smoke.yaml

WORKDIR /app
COPY pyproject.toml README.md ./
COPY keyvision ./keyvision
COPY configs ./configs
COPY app.py ./
RUN pip install --no-cache-dir ".[demo,deploy]"

EXPOSE 7860
CMD ["python", "app.py"]

