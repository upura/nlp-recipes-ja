import tensorflow_hub as hub
import tensorflow_text  # noqa

from utils_nlp.dataset.livedoor import load_pandas_df


if __name__ == '__main__':
    df = load_pandas_df(nrows=10)
    text = df['text'][0][:30]
    print(text)
    # 友人代表のスピーチ、独女はどうこなしている？もうすぐジューン

    embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder-multilingual/3')
    vectors = embed([text])
    print(vectors[0].shape)
    print(vectors[0])
    """
    (512,)
    tf.Tensor(
    [-4.53309491e-02 -5.73447756e-02  3.41094285e-02  1.09533397e-02
    -2.55712979e-02 -8.29478130e-02  3.02479346e-03  8.89975950e-02], shape=(512,), dtype=float32
    """
