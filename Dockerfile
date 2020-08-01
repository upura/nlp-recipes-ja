FROM gcr.io/kaggle-images/python:v76

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
COPY requirements.txt .

RUN pip install -U pip && \
    pip install -r requirements.txt
