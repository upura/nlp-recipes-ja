import neologdn

from utils_nlp.dataset.livedoor import load_pandas_df


if __name__ == '__main__':
    df = load_pandas_df(nrows=10)
    df['text'] = df['text'].apply(neologdn.normalize)
    print(df.head())
