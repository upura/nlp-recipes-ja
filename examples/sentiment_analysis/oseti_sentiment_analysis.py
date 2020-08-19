import oseti

from utils_nlp.dataset.livedoor import load_pandas_df


if __name__ == '__main__':
    df = load_pandas_df(nrows=10)
    text = df['text'][0][:30]
    print(text)
    # 友人代表のスピーチ、独女はどうこなしている？もうすぐジューン

    analyzer = oseti.Analyzer()
    print(analyzer.analyze(text))
    print(analyzer.count_polarity(text))
    # [1.0, 0]
    # [{'positive': 2, 'negative': 0}, {'positive': 0, 'negative': 0}]
