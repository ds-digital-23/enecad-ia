FROM python:3.11-bullseye

ENV PYTHONUNBUFFERED=1

RUN apt-get update -y && apt-get install -y \
    build-essential \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    && apt-get clean

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip --default-timeout=6000

COPY ./requirements.txt /

RUN pip install --default-timeout=6000 -r /requirements.txt --no-cache-dir

COPY ./app /app

WORKDIR /app


CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
