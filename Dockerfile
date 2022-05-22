FROM python:3.9

COPY ./requirements.txt ./
RUN pip install -U pip && pip install -r requirements.txt

RUN apt update
RUN apt install -y osmium-tool \
    && apt install -y libosmium-dev \
    && apt install -y tzdata \
    && apt install -y vim

WORKDIR /usr/src

ENV TZ Asia/Tokyo
ENV PYTHONPATH /usr/src

ENTRYPOINT "/bin/bash"
