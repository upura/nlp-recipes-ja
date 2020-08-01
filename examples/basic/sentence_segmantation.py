import functools

from ja_sentence_segmenter.common.pipeline import make_pipeline
from ja_sentence_segmenter.concatenate.simple_concatenator import concatenate_matching
from ja_sentence_segmenter.normalize.neologd_normalizer import normalize
from ja_sentence_segmenter.split.simple_splitter import split_newline, split_punctuation

from utils_nlp.dataset.livedoor import load_pandas_df


if __name__ == '__main__':
    df = load_pandas_df(nrows=10)

    split_punc2 = functools.partial(split_punctuation, punctuations=r"。!?")
    concat_tail_no = functools.partial(concatenate_matching, former_matching_rule=r"^(?P<result>.+)(の)$", remove_former_matched=False)
    segmenter = make_pipeline(normalize, split_newline, concat_tail_no, split_punc2)

    df['sentences'] = df['text'].apply(lambda x: list(segmenter(x)))
    print(df['sentences'][0])
