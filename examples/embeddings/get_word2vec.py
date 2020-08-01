import nagisa

from utils_nlp.dataset.livedoor import load_pandas_df
from utils_nlp.models.pretrained_embeddings.word2vec import load_pretrained_vectors, convert_to_wv


if __name__ == '__main__':
    df = load_pandas_df(nrows=10)
    text = df['text'][0][:30]
    print(text)
    # 友人代表のスピーチ、独女はどうこなしている？もうすぐジューン

    tagger = nagisa.Tagger()
    nouns = tagger.extract(text, extract_postags=['名詞']).words
    print(nouns)
    # ['友人', '代表', 'スピーチ', '独女', 'ジューン']

    word_vec = load_pretrained_vectors('data')
    vectors = convert_to_wv(nouns[0], word_vec)
    print(vectors.shape)
    # (300,)
    print(vectors[:5])
    # [ 1.0028e-01  1.0647e-02 -1.7439e-01 -2.7110e-03  2.1647e-01]
