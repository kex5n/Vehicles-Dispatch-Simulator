FROM python:3.9 AS experiment

COPY ./requirements.txt ./
RUN pip install -U pip && pip install -r requirements.txt

RUN apt update
RUN apt install -y osmium-tool \
    && apt install -y libosmium-dev \
    && apt install -y tzdata

WORKDIR /usr/src

ENV TZ Asia/Tokyo
ENV PYTHONPATH /usr/src

FROM experiment AS dev

RUN apt install -y vim
COPY ./requirements-dev.txt ./
RUN pip install -r requirements-dev.txt
