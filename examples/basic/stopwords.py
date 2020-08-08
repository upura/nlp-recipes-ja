from collections import Counter
import itertools
from konoha import WordTokenizer
import numpy as np
import pandas as pd

from utils_nlp.dataset.livedoor import load_pandas_df


def remove_stopwords(words, stopwords):
    words = [word for word in words if word not in stopwords]
    return words


def get_stop_words_by_freq(docs, n=100):
    docs = list(itertools.chain(*list(docs)))
    fdist = Counter(docs)
    stopwords = [word for word, freq in fdist.most_common(n)]
    return stopwords


def get_stop_words_by_dict():
    stopwords = pd.read_table('http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt', header=None)
    stopwords = list(stopwords[0].values)
    return stopwords


if __name__ == '__main__':
    df = load_pandas_df(nrows=100)
    tokenizer = WordTokenizer('MeCab')
    docs = np.array([
        map(str, tokenizer.tokenize(text)) for text in df['text']
    ])
    stopwords_f = get_stop_words_by_freq(docs, n=100)
    stopwords_d = get_stop_words_by_dict()
    stopwords = set(stopwords_f) | set(stopwords_d)
    print(stopwords)
    docs = remove_stopwords(docs, stopwords)
