from konoha import WordTokenizer

from utils_nlp.dataset.livedoor import load_pandas_df


if __name__ == '__main__':
    df = load_pandas_df(nrows=10)
    text = df['text'][0][:30]
    print(text)
    # 友人代表のスピーチ、独女はどうこなしている？もうすぐジューン

    tokenizer_m = WordTokenizer('MeCab')
    print(tokenizer_m.tokenize(text))
    # [友人, 代表, の, スピーチ, 、, 独, 女, は, どう, こなし, て, いる, ？, もうすぐ, ジューン]

    tokenizer_s = WordTokenizer('Sudachi', mode='A', with_postag=True)
    print(tokenizer_s.tokenize(text))
    # [友人 (名詞), 代表 (名詞), の (助詞), スピーチ (名詞), 、 (補助記号), 独女 (名詞), は (助詞), どう (副詞), こなし (動詞), て (助詞), いる (動詞), ？ (補助記号), もう (副詞), すぐ (副詞), ジューン (名詞)]

    df['sep_text'] = [tokenizer_m.tokenize(text) for text in df['text']]
    print(df.head())
