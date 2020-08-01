import glob
import os
import tarfile
from urllib.request import urlretrieve

import pandas as pd


def load_pandas_df(nrows: int = None, shuffle: bool = False) -> pd.DataFrame:
    """Loads the livedoor dataset as pd.DataFrame
    This code is from https://github.com/yoheikikuta/bert-japanese/blob/master/notebook/finetune-to-livedoor-corpus.ipynb

    Args:
        nrows (int, optional): [description]. Defaults to None.

    Returns:
        pd.DataFrame: livedoor dataset
    """
    if os.path.exists('./data/livedoor.csv'):
        df = pd.read_csv('./data/livedoor.csv')
    else:
        df = download_livedoor()

    if shuffle:
        df = df.sample(frac=1, random_state=7).reset_index(drop=True)

    if nrows:
        df = df[:nrows]

    return df


def download_livedoor() -> pd.DataFrame:
    """Download the dataset from "https://www.rondhuit.com/download.html", unzip, and load

    Returns:
        pd.DataFrame: livedoor dataset
    """
    FILEURL = 'https://www.rondhuit.com/download/ldcc-20140209.tar.gz'
    FILEPATH = './data/ldcc-20140209.tar.gz'
    EXTRACTDIR = './data/livedoor/'
    urlretrieve(FILEURL, FILEPATH)

    mode = "r:gz"
    tar = tarfile.open(FILEPATH, mode)
    tar.extractall(EXTRACTDIR)
    tar.close()

    categories = [
        name for name
        in os.listdir(os.path.join(EXTRACTDIR, "text"))
        if os.path.isdir(os.path.join(EXTRACTDIR, "text", name))]

    categories = sorted(categories)
    table = str.maketrans({
        '\n': '',
        '\t': '　',
        '\r': '',
    })

    all_text = []
    all_label = []

    for cat in categories:
        files = glob.glob(os.path.join(EXTRACTDIR, "text", cat, "{}*.txt".format(cat)))
        files = sorted(files)
        body = [extract_txt(elem).translate(table) for elem in files]
        label = [cat] * len(body)

        all_text.extend(body)
        all_label.extend(label)

    df = pd.DataFrame({'text': all_text, 'label': all_label})
    df.to_csv('./data/livedoor.csv', index=False)
    return df


def extract_txt(filename: str) -> str:
    with open(filename) as text_file:
        # 0: URL, 1: timestamp
        text = text_file.readlines()[2:]
        text = [sentence.strip() for sentence in text]
        text = list(filter(lambda line: line != '', text))
        return ''.join(text)
