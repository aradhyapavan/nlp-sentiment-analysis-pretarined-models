# syntax=docker/dockerfile:1

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860 \
    HF_HOME=/app/.cache/huggingface \
    NLTK_DATA=/app/nltk_data \
    MPLCONFIGDIR=/app/.config/matplotlib

# System deps (build tools and libs for pillow/wordcloud)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better layer caching)
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Prepare writable caches
RUN mkdir -p ${HF_HOME} ${NLTK_DATA} ${MPLCONFIGDIR}

# Copy application code
COPY . .

# Ensure writable permissions for runtime (Spaces/K8s non-root scenarios)
RUN chmod -R 777 /app

# Run postbuild (e.g., install spaCy model) if present
RUN if [ -f postbuild ]; then sh postbuild; else python -m spacy download en_core_web_md; fi

# Pre-download NLTK data to writable dir
RUN python - <<'PY'
import nltk, os
os.makedirs(os.environ.get('NLTK_DATA','/app/nltk_data'), exist_ok=True)
for pkg in ['punkt','punkt_tab','wordnet','averaged_perceptron_tagger']:
    try:
        nltk.download(pkg, download_dir=os.environ['NLTK_DATA'])
    except Exception as e:
        print('NLTK download failed for', pkg, e)
PY

# Preload HF transformer models to writable cache
RUN python - <<'PY'
from transformers import pipeline
# DistilBERT SST-2
pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
# RoBERTa Twitter
pipeline('sentiment-analysis', model='cardiffnlp/twitter-roberta-base-sentiment')
# Emotion model
pipeline('sentiment-analysis', model='j-hartmann/emotion-english-distilroberta-base')
PY

# Expose default port (can be overridden by $PORT)
EXPOSE 7860

# Start the app using gunicorn (respects $PORT)
CMD ["sh", "-c", "gunicorn -b 0.0.0.0:${PORT:-7860} app:app"]
