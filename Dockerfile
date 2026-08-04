FROM python:3.13-slim

WORKDIR /artifact

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-lock.txt .
RUN python -m pip install --no-cache-dir -r requirements-lock.txt

COPY . .

CMD ["bash", "scripts/run_reproducibility_smoke.sh"]
