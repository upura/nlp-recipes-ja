import nagisa

from utils_nlp.dataset.livedoor import load_pandas_df


if __name__ == '__main__':
    df = load_pandas_df(nrows=10)
    text = df['text'][0][:30]
    print(text)
    # 友人代表のスピーチ、独女はどうこなしている？もうすぐジューン

    tagger = nagisa.Tagger()
    print(tagger.extract(text, extract_postags=['名詞']))
    # 友人/名詞 代表/名詞 スピーチ/名詞 独女/名詞 ジューン/名詞

    df['sep_text'] = [tagger.extract(text, extract_postags=['名詞']).words for text in df['text']]
    print(df.head())
