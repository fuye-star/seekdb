FROM python:3.12-slim

RUN pip install --no-cache-dir numpy==2.5.2

WORKDIR /app
COPY minivectordb.py /app/minivectordb.py

ENTRYPOINT ["python3", "-u", "/app/minivectordb.py"]
