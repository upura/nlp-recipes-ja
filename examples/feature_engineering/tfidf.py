from konoha import WordTokenizer
import neologdn
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from utils_nlp.dataset.livedoor import load_pandas_df


if __name__ == '__main__':
    df = load_pandas_df(nrows=10)

    # Normalization
    df['text'] = df['text'].apply(neologdn.normalize)

    tokenizer = WordTokenizer('MeCab')
    docs = np.array([
        ' '.join(map(str, tokenizer.tokenize(text))) for text in df['text']
    ])
    print(docs.shape)
    # (10,)

    count_vec = CountVectorizer(min_df=2,
                                max_features=20000,
                                ngram_range=(1, 3))
    bags = count_vec.fit_transform(docs)

    print(bags.toarray().shape)
    print(bags.toarray())
    """
    (10, 445)
    [[1 0 1 ... 0 0 0]
    [1 0 0 ... 0 0 0]
    [1 0 0 ... 1 0 0]
    ...
    [0 0 1 ... 0 0 0]
    [0 0 0 ... 0 0 0]
    [0 0 0 ... 0 0 0]]
    """

    tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
    tf_idf = tfidf.fit_transform(bags)
    print(tf_idf.toarray().shape)
    print(tf_idf.toarray())
    """
    (10, 445)
    [[0.04752833 0.05432543 0.         ... 0.         0.         0.        ]
    [0.0484923  0.         0.         ... 0.         0.         0.        ]
    [0.04909543 0.         0.04364936 ... 0.         0.         0.05611665]
    ...
    [0.         0.03772958 0.         ... 0.03772958 0.03772958 0.        ]
    [0.         0.         0.03994261 ... 0.         0.         0.        ]
    [0.         0.         0.         ... 0.         0.         0.        ]]
    """
