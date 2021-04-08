import sys

import numpy as np
import pytorch_lightning as pl
import torch
from transformers import BertModel

sys.path.append('.')
from utils_nlp.eval.classification import eval_classification


class BertClassifier(torch.nn.Module):
    def __init__(self, model_name: str, num_classes: int = 9):
        super().__init__()
        self.l1 = BertModel.from_pretrained(model_name)
        self.l2 = torch.nn.Dropout(0.3)
        if 'large' in model_name:
            self.l3 = torch.nn.Linear(1024, num_classes)
        else:
            self.l3 = torch.nn.Linear(768, num_classes)

    def forward(self, ids, attention_mask, token_type_ids):
        _, output_1 = self.l1(ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=False)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output


class PLBertClassifier(pl.LightningModule):
    def __init__(self, model_name: str, num_classes: int = 9):
        super().__init__()
        self.backbone = BertClassifier(model_name, num_classes)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, ids, attention_mask, token_type_ids):
        output = self.backbone(ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return output

    def training_step(self, batch, batch_idx):
        ids = batch["ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]
        targets = batch["targets"]
        output = self.backbone(ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        loss = self.criterion(output, targets)
        return loss

    def validation_step(self, batch, batch_idx):
        ids = batch["ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]
        targets = batch["targets"]
        output = self.backbone(ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        loss = self.criterion(output, targets)
        return {'oof': output, 'targets': targets}

    def validation_epoch_end(self, outputs):
        oof = np.concatenate(
            [x['oof'].detach().cpu().numpy() for x in outputs], axis=0
        )
        targets = np.concatenate(
            [x['targets'].detach().cpu().numpy() for x in outputs], axis=0
        )
        print(eval_classification(targets, oof.argmax(axis=1)))

    def test_step(self, batch, batch_idx):
        ids = batch["ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]
        targets = batch["targets"]
        output = self.backbone(ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        loss = self.criterion(output, targets)
        return {'preds': output}

    def test_epoch_end(self, outputs):
        preds = np.concatenate(
            [x['preds'].detach().cpu().numpy() for x in outputs], axis=0
        )
        np.save('data/bert/preds', preds)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return [optimizer], [scheduler]
