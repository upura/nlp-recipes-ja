import neologdn
import numpy as np

from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split, StratifiedKFold
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.tokenization_bert_japanese import BertJapaneseTokenizer

from utils_nlp.common.pytorch_utils import seed_everything
from utils_nlp.dataset.livedoor import load_pandas_df
from utils_nlp.eval.classification import eval_classification
from utils_nlp.models.nn.datasets import LivedoorDataset
from utils_nlp.models.nn.runner import CustomRunner
from utils_nlp.models.nn.models import BERTClass


def preprocess_data(df):
    # split
    df['text'] = df['text'].apply(neologdn.normalize)
    le = preprocessing.LabelEncoder()
    df['label'] = le.fit_transform(df['label'])

    X_train, X_test, y_train, y_test = train_test_split(
        df, df['label'].values, test_size=0.2, random_state=42, stratify=df['label'])

    return X_train.reset_index(drop=True), X_test.reset_index(drop=True), y_train, y_test


if __name__ == '__main__':

    RUN_NAME = 'bert'
    MAX_LEN = 20
    seed_everything()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    df = load_pandas_df(nrows=1000, shuffle=True)
    X_train, X_test, y_train, y_test = preprocess_data(df)

    tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')

    test_dataset = LivedoorDataset(X_test, tokenizer, MAX_LEN)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=32)

    y_preds = []
    NUM_CLASS = 9
    oof_train = np.zeros((len(X_train), NUM_CLASS))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    for fold_id, (train_index, valid_index) in enumerate(tqdm(cv.split(X_train, X_train['label']))):

        X_tr = X_train.loc[train_index, :].reset_index(drop=True)
        X_val = X_train.loc[valid_index, :].reset_index(drop=True)
        y_tr = y_train[train_index]
        y_val = y_train[valid_index]

        train_dataset = LivedoorDataset(X_tr, tokenizer, MAX_LEN)
        valid_dataset = LivedoorDataset(X_val, tokenizer, MAX_LEN)

        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=32)
        valid_loader = DataLoader(valid_dataset, shuffle=False, batch_size=32)

        loaders = {'train': train_loader, 'valid': valid_loader}
        runner = CustomRunner(device=device)

        model = BERTClass(NUM_CLASS)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)

        logdir = f'data/{RUN_NAME}/logdir_{RUN_NAME}/fold{fold_id}'
        runner.train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            loaders=loaders,
            logdir=logdir,
            num_epochs=3,
            verbose=True,
        )

        pred = np.concatenate(list(map(lambda x: x.cpu().numpy(),
                                       runner.predict_loader(
                                           loader=valid_loader,
                                           resume=f'{logdir}/checkpoints/best.pth',
                                           model=model,),)))

        oof_train[valid_index] = pred
        score = log_loss(y_val, oof_train[valid_index])
        print('score', score)

        y_pred = np.concatenate(list(map(lambda x: x.cpu().numpy(),
                                         runner.predict_loader(
                                         loader=test_loader,
                                         resume=f'{logdir}/checkpoints/best.pth',
                                         model=model,),)))
        y_preds.append(y_pred)

    y_preds = np.mean(y_preds, axis=0)
    print(f'test, log_loss: {log_loss(y_test, y_preds)}')
    result_dict = eval_classification(y_test, y_preds.argmax(axis=1))
    print(result_dict)
