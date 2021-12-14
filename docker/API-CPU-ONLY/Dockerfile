FROM ubuntu:18.04

LABEL maintainer="Yanis Labrak <yanis.labrak@univ-avignon.fr>"
LABEL repository="qanastek/hugsvision-api-cpu-only"
LABEL version="1.0"
LABEL description="HugsVision docker image with CPU-only PyTorch API backend for HAM10000 Skin Cancer."

LABEL multi.label1="HAM10000" multi.label2="API" other="cpu-only"

WORKDIR /app

RUN apt update && \
    apt install -y bash \
                   build-essential \
                   git \
                   curl \
                   ca-certificates \
                   python3.8 \
                   python3-pip \
                   python3.8-dev \
                   ffmpeg \
                   libsm6 \
                   libxext6

RUN rm -rf /var/lib/apt/lists

RUN ln -sf /usr/bin/python3.8 /usr/bin/python
RUN ln -sf /usr/bin/python3.8 /usr/bin/python3

RUN python3 -m pip install --no-cache-dir --upgrade pip

RUN python3 -m pip install --no-cache-dir \
    jupyter \
    cython \
    torch \
    numpy

RUN python3 -m pip install  --no-cache-dir \
    fastapi \ 
    uvicorn \ 
    python-multipart

RUN python3 -m pip install  --no-cache-dir \
    hugsvision==0.75

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

COPY ./server /app
COPY ./README.md /app/README.md
RUN cd /app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]

EXPOSE 80
