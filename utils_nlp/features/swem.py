import numpy as np
from tqdm import tqdm

from utils_nlp.models.pretrained_embeddings.word2vec import convert_to_wv


def create(docs, word_vec, aggregation='max'):
    """Create swem vectors

    Args:
        docs: np.array()
        word_vec: Loaded word2vectors
        aggregation (str, optional): How to do max-pooling, 'max' or 'mean'. Defaults to 'max'.

    Raises:
        ValueError: Invalid aggregation arg

    Returns:
        swem: Created swem vectors
    """
    if aggregation == 'max':
        agg = np.max
    elif aggregation == 'mean':
        agg = np.mean
    else:
        raise ValueError()

    swem = []
    for sentence in tqdm(docs, total=len(docs)):
        embed_i = [convert_to_wv(s, word_vec) for s in sentence]
        embed_i = np.array(embed_i)
        embed_i = agg(embed_i, axis=0)
        swem.append(embed_i)
    swem = np.array(swem)
    return swem
