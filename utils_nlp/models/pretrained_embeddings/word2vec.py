import os
import requests
import zipfile

import numpy as np
from tqdm import tqdm
import gensim


def _maybe_download_and_extract(dest_path, file_name):
    """ Downloads and extracts Word2vec vectors if they don’t already exist
    Args:
        dest_path: Path to the directory where the vectors will be extracted.
        file_name: File name of the word2vec vector file.
    Returns:
         str: File path to the word2vec vector file.
    """

    dir_path = os.path.join(dest_path, "word2vec")
    file_path = os.path.join(dir_path, file_name)
    dl_path = os.path.join(file_path, '{}.zip'.format(file_name))

    if not os.path.exists(file_path):
        os.makedirs(file_path, exist_ok=True)
        download_from_gdrive('0ByFQ96A4DgSPUm9wVWRLdm5qbmc', destination=dl_path)
        with zipfile.ZipFile(dl_path) as f:
            f.extractall(file_path)
    else:
        print("Vector file already exists. No changes made.")

    return file_path


def load_pretrained_vectors(
    dir_path, file_name='vector_neologd', limit=None
):
    """ Method that loads word2vec vectors. Downloads if it doesn't exist.
    Args:
        file_name(str): Name of the word2vec file.
        dir_path(str): Path to the directory where word2vec vectors exist or will be
        downloaded.
        limit(int): Number of word vectors that is loaded from gensim. This option
        allows us to save RAM space and avoid memory errors.
    Returns:
        gensim.models.keyedvectors.Word2VecKeyedVectors: Loaded word2vectors
    """
    file_path = _maybe_download_and_extract(dir_path, file_name)
    model_path = os.path.join(file_path, 'model.vec')
    word2vec_vectors = gensim.models.KeyedVectors.load_word2vec_format(
        model_path, binary=False, limit=limit
    )

    return word2vec_vectors


def download_from_gdrive(id, destination):
    """
    Download file from Google Drive
    :param str id: g-drive id
    :param str destination: output path
    :return:
    """
    url = "https://docs.google.com/uc?export=download"

    session = requests.Session()
    response = session.get(url, params={'id': id}, stream=True)
    token = get_confirm_token(response)
    if token:
        print("get download warning. set confirm token.")
        params = {'id': id, 'confirm': token}
        response = session.get(url, params=params, stream=True)
    save_response_content(response, destination)


def get_confirm_token(response):
    """
    verify whether warned or not.
    [note] In Google Drive Api, if requests content size is large,
    the user are send to verification page.
    :param requests.Response response:
    :return:
    """
    for k, v in response.cookies.items():
        if k.startswith("download_warning"):
            return v

    return None


def save_response_content(response, destination):
    """
    :param requests.Response response:
    :param str destination:
    :return:
    """
    chunk_size = 1024 * 1024
    print("start downloading...")
    with open(destination, "wb") as f:
        for chunk in tqdm(response.iter_content(chunk_size), unit="MB"):
            f.write(chunk)
    print("Finish!!")
    print("Save to:{}".format(destination))


def convert_to_wv(w: str, word_vec):
    """Convert word to vectors

    Args:
        w (str): Word
        word_vec: Loaded word2vectors

    Returns:
        [type]: numpy vectors
    """
    try:
        v = word_vec.word_vec(w)
    except KeyError:
        v = np.zeros(shape=(word_vec.vector_size,))
    return v
