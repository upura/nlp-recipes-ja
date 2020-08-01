FROM gcr.io/kaggle-images/python:v76

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
COPY requirements.txt .

# mecab
RUN apt-get update -y && \
    apt-get install -y mecab libmecab-dev mecab-ipadic-utf8

RUN pip install -U pip && \
    pip install -r requirements.txt
