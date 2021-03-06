# NLP Recipes for Japanese

This repository contains samples codes for natural language processing in Japanese.
It's highly inspired by [microsoft/nlp-recipes](https://github.com/microsoft/nlp-recipes).

## Content

The following is a summary of the commonly used NLP scenarios covered in the repository. Each scenario is demonstrated in one or more scripts or Jupyter notebook examples that make use of the core code base of models and repository utilities.

|Category|Methods|
|---| --- |
|[Basic](./examples/basic)|Cleaning, Normalization, Stopwords, Sentence Segmantation, Ruby|
|[Embeddings](./examples/embeddings)|Word2Vec, fastText, Universal Sentence Encoder|
|[Feature Engineering](./examples/feature_engineering)|Bag-of-Words, TF-IDF, BM25, SWEM, SCDV|
|[Morphological Analysis](./examples/morphological_analysis)|Konoha, nagisa|
|[Sentence Similarity](./examples/sentence_similarity)|Cosine Similarity|
|[Sentiment Analysis](sentiment_analysis)|oseti|
|[Text Classification](./examples/text_classification)|TF-IDF & Logistic Regression, TF-IDF & LightGBM, BERT, T5|
|[Visualization](./examples/visualization)|Visualization with Japanese texts|

## Environment

```bash
docker-compose up -d --build
docker exec -it nlp-recipes-ja bash
```
