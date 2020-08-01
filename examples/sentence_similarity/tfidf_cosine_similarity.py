from konoha import WordTokenizer
import neologdn
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity

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

    print(cosine_similarity(tf_idf.toarray()))
    """
    [[1.         0.31294546 0.22234506 0.27272853 0.22658861 0.37452715
    0.35456225 0.29524085 0.17193537 0.36229732]
    [0.31294546 1.         0.25102573 0.25264431 0.24334397 0.33785512
    0.31670052 0.28218417 0.12684395 0.32628839]
    [0.22234506 0.25102573 1.         0.24099022 0.17307931 0.31050187
    0.32489792 0.28119098 0.15070305 0.38326419]
    [0.27272853 0.25264431 0.24099022 1.         0.23456837 0.32640547
    0.27615115 0.3153026  0.26716363 0.31163735]
    [0.22658861 0.24334397 0.17307931 0.23456837 1.         0.41007705
    0.24911698 0.36058785 0.11835559 0.2387821 ]
    [0.37452715 0.33785512 0.31050187 0.32640547 0.41007705 1.
    0.45739635 0.32316926 0.2059866  0.31257367]
    [0.35456225 0.31670052 0.32489792 0.27615115 0.24911698 0.45739635
    1.         0.39132051 0.24839521 0.3321967 ]
    [0.29524085 0.28218417 0.28119098 0.3153026  0.36058785 0.32316926
    0.39132051 1.         0.15238316 0.30832032]
    [0.17193537 0.12684395 0.15070305 0.26716363 0.11835559 0.2059866
    0.24839521 0.15238316 1.         0.24724469]
    [0.36229732 0.32628839 0.38326419 0.31163735 0.2387821  0.31257367
    0.3321967  0.30832032 0.24724469 1.        ]]
    """
