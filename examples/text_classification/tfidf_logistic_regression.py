import sys

from konoha import WordTokenizer
from loguru import logger
import neologdn
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split, StratifiedKFold
from tqdm import tqdm

sys.path.append('.')
from utils_nlp.common.data import Data
from utils_nlp.dataset.livedoor import load_pandas_df
from utils_nlp.eval.classification import eval_classification


def preprocess_data(df):
    # split
    df['text'] = df['text'].apply(neologdn.normalize)
    le = preprocessing.LabelEncoder()
    df['label'] = le.fit_transform(df['label'])

    df_train, df_test, y_train, y_test = train_test_split(
        df, df['label'].values, test_size=0.2, random_state=42, stratify=df['label'])

    # tokenize
    tokenizer = WordTokenizer('MeCab')
    docs_train = np.array([
        ' '.join(map(str, tokenizer.tokenize(text))) for text in df_train['text']
    ])
    docs_test = np.array([
        ' '.join(map(str, tokenizer.tokenize(text))) for text in df_test['text']
    ])

    # tfidf: Don't use df_test for fitting
    count_vec = CountVectorizer(min_df=2,
                                max_features=20000,
                                ngram_range=(1, 3))
    bags_train = count_vec.fit_transform(docs_train)
    bags_test = count_vec.transform(docs_test)

    tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
    tf_idf_train = tfidf.fit_transform(bags_train)
    tf_idf_test = tfidf.transform(bags_test)

    X_train = pd.DataFrame(tf_idf_train.toarray())
    X_test = pd.DataFrame(tf_idf_test.toarray())

    return X_train.reset_index(drop=True), X_test.reset_index(drop=True), y_train, y_test


if __name__ == '__main__':

    df = load_pandas_df(shuffle=True)
    X_train, X_test, y_train, y_test = preprocess_data(df)

    RUN_NAME = 'logistic_regression'
    logger.add(f'data/{RUN_NAME}/result.log',
               colorize=True,
               format='<green>{time}</green> {message}')
    logger.info(f'{X_train.shape}, {X_test.shape}')

    y_preds = []
    NUM_CLASS = 9
    oof_train = np.zeros((len(X_train), NUM_CLASS))
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)

    for fold_id, (train_index, valid_index) in enumerate(tqdm(cv.split(X_train, y_train))):
        if fold_id == 0:
            X_tr = X_train.loc[train_index, :]
            X_val = X_train.loc[valid_index, :]
            y_tr = y_train[train_index]
            y_val = y_train[valid_index]

            model = LogisticRegression(penalty='l2', solver='sag', random_state=0)
            model.fit(X_tr, y_tr)
            Data.dump(model, f'data/{RUN_NAME}/model_{fold_id}.pkl')

            oof_train[valid_index] = model.predict_proba(X_val)
            score = log_loss(y_val, oof_train[valid_index])
            logger.info(f'fold {fold_id}, log_loss: {score}')

            y_pred = model.predict_proba(X_test)
            y_preds.append(y_pred)

    y_preds = np.mean(y_preds, axis=0)
    logger.info(f'test, log_loss: {log_loss(y_test, y_preds)}')
    result_dict = eval_classification(y_test, y_preds.argmax(axis=1))
    logger.info(str(result_dict))
    """
    {'accuracy': 0.9308,
     'precision': [0.8771, 0.96, 0.9639, 0.9412, 0.9198, 0.8678, 0.9771, 0.9309, 0.9517],
     'recall': [0.9023, 0.9655, 0.9249, 0.7843, 0.9885, 0.8935, 0.9828, 0.9722, 0.8961],
     'f1': [0.8895, 0.9628, 0.944, 0.8556, 0.9529, 0.8805, 0.9799, 0.9511, 0.9231]}
    """

    Data.dump(oof_train, f'data/{RUN_NAME}/oof_train.pkl')
    Data.dump(y_preds, f'data/{RUN_NAME}/y_preds.pkl')
