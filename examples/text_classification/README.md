# Text Classification

This folder contains examples of text classification models.

## What is Text Classification?

>Text classification is a supervised learning method of learning and predicting the category or the class of a document given its text content.
>The state-of-the-art methods are based on neural networks of different architectures as well as pre-trained language models or word embeddings.

https://github.com/microsoft/nlp-recipes/blob/master/examples/text_classification/README.md

## Summary

|Notebook|Environment|Description|ACC|
|---|---|---|---|
|[TF-IDF & Logistic Regression](tfidf_logistic_regression.py)|Local| [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) with TF-IDF vectors | 0.9308 |
|[TF-IDF & LightGBM](tfidf_lgbm.py)|Local| [LightGBM](https://github.com/microsoft/LightGBM) with TF-IDF vectors | 0.9512 |
|[BERT](run_bert.py) 'cl-tohoku/bert-base-japanese-v2' |Local| [Transformers BERT](https://github.com/huggingface/transformers) | 0.9362 |
|[BERT](run_bert.py) 'cl-tohoku/bert-base-japanese-char-v2' |Local| [Transformers BERT](https://github.com/huggingface/transformers) | 0.9274 |

Accuracy scores (ACC) are calculated by running code only in fold 0 in the condition that datasets are devided into train/val/test at the rate of 0.6/0.2/0.2.
