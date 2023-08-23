# FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel
FROM python:3.10

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
COPY requirements.txt .

# mecab
RUN apt update -y && \
    apt install -y mecab libmecab-dev mecab-ipadic-utf8

RUN pip install -U pip && \
    pip install -r requirements.txt
