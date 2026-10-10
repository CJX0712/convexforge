# ConvexForge — reproducible CPU image (author: 晨星)
FROM python:3.13-slim

LABEL org.opencontainers.image.title="ConvexForge" \
      org.opencontainers.image.description="KKT-certified convex optimization suite" \
      org.opencontainers.image.authors="晨星" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.version="0.1.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1

WORKDIR /app

COPY requirements.lock.txt ./
RUN pip install --no-cache-dir -r requirements.lock.txt

COPY pyproject.toml README.md LICENSE ./
COPY convexforge ./convexforge
COPY tests ./tests

RUN pip install --no-cache-dir --no-deps -e .

# Sanity: the image must pass the invariant gates out of the box.
RUN python -m convexforge.cli selftest

ENTRYPOINT ["python", "-m", "convexforge.cli"]
CMD ["demo"]
