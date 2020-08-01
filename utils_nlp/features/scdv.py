from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


def create(docs, word_vec, n_components=10):
    """Create scdv vectors

    Args:
        docs: np.array()
        word_vec: Loaded word2vectors
        n_components (int, optional): Number of components

    Returns:
        swem: Created scdv vectors
    """
    n_wv_embed = word_vec.vector_size

    # Create vocab set of w2v model and corpus
    vocab_model = set(k for k in word_vec.vocab.keys())
    vocab_docs = set([w for doc in docs for w in doc])
    out_of_vocabs = len(vocab_docs) - len(vocab_docs & vocab_model)
    print('out of vocabs: {out_of_vocabs}'.format(**locals()))
    use_words = list(vocab_docs & vocab_model)

    df_use = pd.DataFrame()
    df_use['word'] = use_words
    df_idf = create_idf_dataframe(docs)
    df_use = pd.merge(df_use, df_idf, on='word', how='left')
    idf = df_use['idf'].values

    use_word_vectors = np.array([word_vec[w] for w in use_words])

    clf = GaussianMixture(n_components=n_components, covariance_type='tied', verbose=2)
    clf.fit(use_word_vectors)

    word_probs = clf.predict_proba(use_word_vectors)
    # (n_vocabs, n_components,)
    word_cluster_vector = use_word_vectors[:, None, :] * word_probs[:, :, None]
    # (n_vocabs, n_components, n_wv_embed)

    topic_vector = word_cluster_vector.reshape(-1, n_components * n_wv_embed) * idf[:, None]

    topic_vector[np.isnan(topic_vector)] = 0
    word_to_topic = dict(zip(use_words, topic_vector))
    n_embedding = topic_vector.shape[1]

    cdv_vector = create_document_vector(docs, word_to_topic, n_embedding)
    compressed = compress_document_vector(cdv_vector)

    return compressed


def create_idf_dataframe(documents):
    """Create idf pd.DataFrame

    Args:
        documents (list[str]):
    Returns:
        [pd.DataFrame]: Created pd.DataFrame
    """

    d = defaultdict(int)

    for doc in documents:
        vocab_i = set(doc)
        for w in list(vocab_i):
            d[w] += 1

    df_idf = pd.DataFrame()
    df_idf['count'] = d.values()
    df_idf['word'] = d.keys()
    df_idf['idf'] = np.log(len(documents) / df_idf['count'])
    return df_idf


def create_document_vector(documents, w2t, n_embedding):
    doc_vectors = []

    for doc in documents:
        vector_i = np.zeros(shape=(n_embedding,))
        for w in doc:
            try:
                v = w2t[w]
                vector_i += v
            except KeyError:
                continue
        doc_vectors.append(vector_i)
    return np.array(doc_vectors)


def compress_document_vector(doc_vector, p=.04):
    v = np.copy(doc_vector)
    vec_norm = np.linalg.norm(v, axis=1)
    # To escape from zero division
    vec_norm = np.where(vec_norm > 0, vec_norm, 1.)
    v /= vec_norm[:, None]

    a_min = v.min(axis=1).mean()
    a_max = v.max(axis=1).mean()
    threshold = (abs(a_min) + abs(a_max)) / 2. * p
    v[abs(v) < threshold] = .0
    return v
