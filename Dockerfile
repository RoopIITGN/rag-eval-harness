# CPU-only build. No GPU is used anywhere in this project, and the default
# PyTorch wheel bundles CUDA -- roughly 2.5GB unpacked, for nothing.
FROM python:3.12-slim

WORKDIR /app

# Dependencies first, so a code change doesn't reinstall torch.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir "fastapi>=0.115" "uvicorn[standard]>=0.32"

# The cross-encoder is downloaded at import time on first use. Baking it into
# the image keeps container start-up off the Hugging Face Hub -- which matters
# on a platform that scales to zero and cold-starts on demand.
RUN python -c "from sentence_transformers import CrossEncoder; \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cpu')"

COPY src/ ./src/

# Credentials come from the platform, never the image.
ENV PYTHONPATH=/app/src PYTHONUNBUFFERED=1
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://localhost:8000/healthz').status==200 else 1)"

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
