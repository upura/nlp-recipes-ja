from konoha import WordTokenizer
import neologdn
import numpy as np

from utils_nlp.dataset.livedoor import load_pandas_df
from utils_nlp.features import swem
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
    swem_max = swem.create(docs, word_vec, aggregation='max')
    swem_mean = swem.create(docs, word_vec, aggregation='mean')
    print(swem_max.shape)
    # (10, 300)
