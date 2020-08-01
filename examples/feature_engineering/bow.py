from konoha import WordTokenizer
import neologdn
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

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
    # (10, 445)
    print(bags.toarray())
    """
    [[1 0 1 ... 0 0 0]
    [1 0 0 ... 0 0 0]
    [1 0 0 ... 1 0 0]
    ...
    [0 0 1 ... 0 0 0]
    [0 0 0 ... 0 0 0]
    [0 0 0 ... 0 0 0]]
    """
