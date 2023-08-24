import random
import sys
import tempfile
from pathlib import Path

import hydra
import mlflow
import numpy as np
import torch
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BatchEncoding,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import set_seed

HOME_PATH = Path(__file__).resolve().parents[2]
try:
    sys.path.append(str(HOME_PATH))
    from utils.mlflow_utils import log_all_params
except Exception as e:
    raise e


def preprocess_text_classification(
    example: dict[str, str | int], tokenizer: AutoTokenizer
) -> BatchEncoding:
    encoded_example = tokenizer(
        example["text"], max_length=512, padding="max_length", truncation=True
    )
    encoded_example["labels"] = example["label"]
    return encoded_example


def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    """予測ラベルと正解ラベルから正解率を計算"""
    softmax = torch.nn.Softmax(1)
    predictions, labels = eval_pred
    outputs = softmax(torch.Tensor(predictions))
    proba = outputs[:, 1]
    predictions = np.argmax(predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    auc = roc_auc_score(labels, proba)
    return {
        "accuracy": (predictions == labels).mean(),
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "auc": auc,
    }


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    set_seed(cfg.seed)
    random.seed(cfg.seed)

    # mlflow setting
    mlflow_tags, mlflow_params, mlflow_metrics = dict(), dict(), dict()
    tmp_folder = tempfile.TemporaryDirectory()
    mlflow_tmp_folder = Path(tmp_folder.name)
    mlflow.set_tracking_uri(HOME_PATH / "mlruns")
    mlflow.set_experiment(cfg.experiment_name)

    # dataset
    indice = [i for i in range(25000)]
    random.shuffle(indice)
    imdb = load_dataset("imdb")
    train_dataset = imdb["train"].select(indice[:3000])
    valid_dataset = imdb["train"].select(indice[3000:6000])
    test_dataset = imdb["test"].select(indice[:3000])

    # preprocessing
    model_name = str(cfg.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoded_train_dataset = train_dataset.map(
        lambda example: preprocess_text_classification(example, tokenizer),
        remove_columns=train_dataset.column_names,
    )
    encoded_valid_dataset = valid_dataset.map(
        lambda example: preprocess_text_classification(example, tokenizer),
        remove_columns=valid_dataset.column_names,
    )
    encoded_test_dataset = test_dataset.map(
        lambda example: preprocess_text_classification(example, tokenizer),
        remove_columns=test_dataset.column_names,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # model definition
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    training_args = TrainingArguments(output_dir=mlflow_tmp_folder, **cfg.params)
    trainer = Trainer(
        model=model,
        train_dataset=encoded_train_dataset,
        eval_dataset=encoded_valid_dataset,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # evaluation
    train_metrics = trainer.evaluate(encoded_train_dataset, metric_key_prefix="train")
    eval_metrics = trainer.evaluate(encoded_valid_dataset)
    test_metrics = trainer.evaluate(encoded_test_dataset, metric_key_prefix="test")
    mlflow_metrics = train_metrics | eval_metrics | test_metrics

    # save results
    with mlflow.start_run():
        mlflow.set_tags(mlflow_tags)
        mlflow.log_params(mlflow_params)
        log_all_params(cfg)
        mlflow.log_metrics(mlflow_metrics)
        mlflow.log_artifacts(mlflow_tmp_folder)
        components = {
            "model": model,
            "tokenizer": tokenizer,
        }
        mlflow.transformers.log_model(
            transformers_model=components,
            artifact_path="transformers_model",
        )
        tmp_folder.cleanup()


if __name__ == "__main__":
    main()
