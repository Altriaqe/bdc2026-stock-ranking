FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

RUN apt-get update && apt-get install -y --no-install-recommends bash libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv && \
    uv sync --frozen --no-dev --index-url https://mirrors.aliyun.com/pypi/simple/

RUN mkdir -p /opt/bdc2026-bundle/data
COPY data/train.csv data/test.csv /opt/bdc2026-bundle/data/

COPY . .

ENV PATH="/app/.venv/bin:$PATH"

CMD ["bash", "/app/run.sh"]
