import pykakasi

from utils_nlp.dataset.livedoor import load_pandas_df


if __name__ == '__main__':
    df = load_pandas_df(nrows=10)
    text = df['text'][0][:30]
    print(text)

    kakasi = pykakasi.kakasi()
    kakasi.setMode("H", "a")        # Hiragana to ascii, default: no conversion
    kakasi.setMode("K", "a")        # Katakana to ascii, default: no conversion
    kakasi.setMode("J", "a")        # Japanese to ascii, default: no conversion
    kakasi.setMode("r", "Hepburn")  # default: use Hepburn Roman table
    kakasi.setMode("s", True)       # add space, default: no separator
    kakasi.setMode("C", True)       # capitalize, default: no capitalize
    conv = kakasi.getConverter()
    result = conv.do(text)
    print(result)
