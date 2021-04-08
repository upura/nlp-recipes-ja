import argparse
import sys

import neologdn
import numpy as np
import pytorch_lightning as pl
from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split, StratifiedKFold
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer

sys.path.append('.')
from utils_nlp.dataset.livedoor import load_pandas_df
from utils_nlp.eval.classification import eval_classification
from utils_nlp.models.nn.datasets import LivedoorDataset
from utils_nlp.models.nn.models import PLBertClassifier


def preprocess_data(df):
    # split
    df['text'] = df['text'].apply(neologdn.normalize)
    le = preprocessing.LabelEncoder()
    df['label'] = le.fit_transform(df['label'])

    X_train, X_test, y_train, y_test = train_test_split(
        df, df['label'].values, test_size=0.2, random_state=42, stratify=df['label'])

    return X_train.reset_index(drop=True), X_test.reset_index(drop=True), y_train, y_test


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name')
    args = parser.parse_args()

    MODEL_NAME = args.model_name
    MAX_LEN = 300
    pl.seed_everything(777)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    df = load_pandas_df(shuffle=True)
    X_train, X_test, y_train, y_test = preprocess_data(df)

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    test_dataset = LivedoorDataset(X_test, tokenizer, MAX_LEN)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=32, num_workers=4)

    y_preds = []
    NUM_CLASS = 9
    oof_train = np.zeros((len(X_train), NUM_CLASS))
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)

    for fold_id, (train_index, valid_index) in enumerate(tqdm(cv.split(X_train, X_train['label']))):
        if fold_id == 0:
            X_tr = X_train.loc[train_index, :].reset_index(drop=True)
            X_val = X_train.loc[valid_index, :].reset_index(drop=True)
            y_tr = y_train[train_index]
            y_val = y_train[valid_index]

            train_dataset = LivedoorDataset(X_tr, tokenizer, MAX_LEN)
            valid_dataset = LivedoorDataset(X_val, tokenizer, MAX_LEN)

            train_loader = DataLoader(train_dataset, shuffle=True, batch_size=16, num_workers=4)
            valid_loader = DataLoader(valid_dataset, shuffle=False, batch_size=32, num_workers=4)

            model = PLBertClassifier(model_name=MODEL_NAME,
                                     num_classes=NUM_CLASS)
            device ='cuda:0' if torch.cuda.is_available() else 'cpu'
            model = model.to(device)
            trainer = pl.Trainer(gpus=1, max_epochs=7)
            trainer.fit(model, train_loader, valid_loader)
            trainer.test(test_dataloaders=test_loader)

    y_preds = np.load('data/bert/preds.npy')
    print(f'test, log_loss: {log_loss(y_test, y_preds)}')
    result_dict = eval_classification(y_test, y_preds.argmax(axis=1))
    print(result_dict)
    """
    {'accuracy': 0.9362,
     'precision': [0.8939, 0.9101, 0.9588, 0.9293, 0.9451, 0.9241, 0.9822, 0.9882, 0.8935],
     'recall': [0.9195, 0.931, 0.9422, 0.902, 0.9885, 0.8639, 0.954, 0.9333, 0.9805],
     'f1': [0.9065, 0.9205, 0.9504, 0.9154, 0.9663, 0.893, 0.9679, 0.96, 0.935]}
    """
