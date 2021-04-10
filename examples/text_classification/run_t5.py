import argparse
import sys
import gc

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
from transformers import T5Tokenizer

sys.path.append('.')
from utils_nlp.dataset.livedoor import load_pandas_df
from utils_nlp.eval.classification import eval_classification
from utils_nlp.models.nn.datasets import LivedoorDatasetT5 as LivedoorDataset
from utils_nlp.models.nn.models import PLT5Classifier


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

    tokenizer = T5Tokenizer.from_pretrained("sonoisa/t5-base-japanese", is_fast=True)

    test_dataset = LivedoorDataset(X_test, tokenizer, MAX_LEN)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1, num_workers=4)

    y_preds = []
    NUM_CLASS = 9
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

            model = PLT5Classifier(model_name=MODEL_NAME)
            device ='cuda:0' if torch.cuda.is_available() else 'cpu'
            model = model.to(device)
            trainer = pl.Trainer(gpus=1, max_epochs=5)
            trainer.fit(model, train_loader, valid_loader)
            model.tokenizer.save_pretrained('data/t5')
            model.backbone.save_pretrained('data/t5')
            del train_loader, train_dataset, valid_loader, valid_dataset, X_train, X_test, y_train, df, X_tr, X_val
            gc.collect()
            trainer.test(test_dataloaders=test_loader)

    y_preds = np.load('data/t5/preds.npy')
    y_preds = np.array([int(d) for d in y_preds])
    result_dict = eval_classification(y_test, y_preds)
    print(result_dict)
    """
    {'accuracy': 0.9566,
     'precision': [0.9699, 0.9194, 0.9815, 0.9583, 0.95, 0.9128, 0.977, 0.9888, 0.956],
     'recall': [0.9253, 0.9828, 0.9191, 0.902, 0.9828, 0.929, 0.977, 0.9833, 0.987],
     'f1': [0.9471, 0.95, 0.9493, 0.9293, 0.9661, 0.9208, 0.977, 0.9861, 0.9712]}
    """
