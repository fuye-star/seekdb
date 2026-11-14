FROM python:3.12-slim

RUN pip install --no-cache-dir numpy==2.5.2 orjson==3.10.7

WORKDIR /app
COPY minivectordb.py /app/minivectordb.py

ENV OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

ENTRYPOINT ["python3", "-u", "/app/minivectordb.py"]
