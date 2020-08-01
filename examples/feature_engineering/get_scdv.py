from konoha import WordTokenizer
import neologdn
import numpy as np

from utils_nlp.dataset.livedoor import load_pandas_df
from utils_nlp.features import scdv
from utils_nlp.models.pretrained_embeddings.word2vec import load_pretrained_vectors


if __name__ == '__main__':
    df = load_pandas_df(nrows=10)

    # Normalization
    df['text'] = df['text'].apply(neologdn.normalize)

    tokenizer = WordTokenizer('MeCab')
    docs = np.array([
        map(str, tokenizer.tokenize(text)) for text in df['text']
    ])
    print(docs.shape)
    # (10,)

    word_vec = load_pretrained_vectors('data')
    scdv = scdv.create(docs, word_vec, n_components=10)
    print(scdv.shape)
    # (10, 3000)
