import matplotlib.pyplot as plt
import japanize_matplotlib

from utils_nlp.dataset.livedoor import load_pandas_df


if __name__ == '__main__':
    df = load_pandas_df()
    df['first_char'] = df['text'].str[0]
    plot_df = df['first_char'].value_counts()[:10].reset_index()

    japanize_matplotlib.japanize()
    plt.figure(figsize=(15, 8))
    plt.bar(plot_df['index'], plot_df['first_char'])
    plt.savefig('examples/visualization/japanize.png')
