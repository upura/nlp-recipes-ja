from konoha import WordTokenizer
import neologdn
import numpy as np
import scipy as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, _document_frequency
from sklearn.utils.validation import check_is_fitted

from utils_nlp.dataset.livedoor import load_pandas_df


class BM25Transformer(BaseEstimator, TransformerMixin):
    """
    Parameters
    ----------
    use_idf : boolean, optional (default=True)
    k1 : float, optional (default=2.0)
    b  : float, optional (default=0.75)
    References
    ----------
    Okapi BM25: a non-binary model - Introduction to Information Retrieval
    http://nlp.stanford.edu/IR-book/html/htmledition/okapi-bm25-a-non-binary-model-1.html
    """
    def __init__(self, use_idf=True, k1=2.0, b=0.75):
        self.use_idf = use_idf
        self.k1 = k1
        self.b = b

    def fit(self, X):
        """
        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features] document-term matrix
        """
        if not sp.sparse.issparse(X):
            X = sp.sparse.csc_matrix(X)
        if self.use_idf:
            n_samples, n_features = X.shape
            df = _document_frequency(X)
            idf = np.log((n_samples - df + 0.5) / (df + 0.5))
            self._idf_diag = sp.sparse.spdiags(idf, diags=0, m=n_features, n=n_features)

        doc_len = X.sum(axis=1)
        self._average_document_len = np.average(doc_len)

        return self

    def transform(self, X, copy=True):
        """
        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features] document-term matrix
        copy : boolean, optional (default=True)
        """
        if hasattr(X, 'dtype') and np.issubdtype(X.dtype, np.floating):
            # preserve float family dtype
            X = sp.sparse.csr_matrix(X, copy=copy)
        else:
            # convert counts or binary occurrences to floats
            X = sp.sparse.csr_matrix(X, dtype=np.float, copy=copy)

        n_samples, n_features = X.shape

        # Document length (number of terms) in each row
        # Shape is (n_samples, 1)
        doc_len = X.sum(axis=1)
        # Number of non-zero elements in each row
        # Shape is (n_samples, )
        sz = X.indptr[1:] - X.indptr[0:-1]

        # In each row, repeat `doc_len` for `sz` times
        # Shape is (sum(sz), )
        # Example
        # -------
        # dl = [4, 5, 6]
        # sz = [1, 2, 3]
        # rep = [4, 5, 5, 6, 6, 6]
        rep = np.repeat(np.asarray(doc_len), sz)

        # Compute BM25 score only for non-zero elements
        nom = self.k1 + 1
        denom = X.data + self.k1 * (1 - self.b + self.b * rep / self._average_document_len)
        data = X.data * nom / denom

        X = sp.sparse.csr_matrix((data, X.indices, X.indptr), shape=X.shape)

        if self.use_idf:
            check_is_fitted(self, '_idf_diag', 'idf vector is not fitted')

            expected_n_features = self._idf_diag.shape[0]
            if n_features != expected_n_features:
                raise ValueError("Input has n_features=%d while the model"
                                 " has been trained with n_features=%d" % (
                                     n_features, expected_n_features))
            X = X * self._idf_diag

        return X


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

    bm25 = BM25Transformer(use_idf=True, k1=2.0, b=0.75)
    bm25 = bm25.fit_transform(bags)
    print(bm25.toarray().shape)
    print(bm25.toarray())
    """
    (10, 445)
    [[0.75499451 1.21230177 0.         ... 0.         0.         0.        ]
    [0.77036179 0.         0.         ... 0.         0.         0.        ]
    [0.83310313 0.         0.40196374 ... 0.         0.         1.3377215 ]
    ...
    [0.         1.02087499 0.         ... 1.02087499 1.02087499 0.        ]
    [0.         0.         0.43242275 ... 0.         0.         0.        ]
    [0.         0.         0.         ... 0.         0.         0.        ]]
    """
