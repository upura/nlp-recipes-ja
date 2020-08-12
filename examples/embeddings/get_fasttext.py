import gensim
import nagisa

from utils_nlp.dataset.livedoor import load_pandas_df


if __name__ == '__main__':
    df = load_pandas_df(nrows=10)
    text = df['text'][0][:30]
    print(text)
    # 友人代表のスピーチ、独女はどうこなしている？もうすぐジューン

    tagger = nagisa.Tagger()
    nouns = tagger.extract(text, extract_postags=['名詞']).words
    print(nouns)
    # ['友人', '代表', 'スピーチ', '独女', 'ジューン']

    model_w = gensim.models.KeyedVectors.load_word2vec_format('./data/fasttext/cc.ja.300.vec.gz', binary=False)
    for noun in nouns:
        try:
            print(noun, model_w[noun].shape)
        except KeyError:
            print(noun, 'Out of vocabulary')
    """
    友人 (300,)
    代表 (300,)
    スピーチ (300,)
    独女 Out of vocabulary
    ジューン (300,)
    """

    model_f = gensim.models.fasttext.load_facebook_model('./data/fasttext/cc.ja.300.bin')
    for noun in nouns:
        print(noun, noun in model_f.wv.vocab)
        print(noun, model_f[noun].shape)
