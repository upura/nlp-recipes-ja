from catalyst.dl import Runner
import torch


class CustomRunner(Runner):
    def _handle_batch(self, batch):
        ids = batch['ids']
        mask = batch['mask']
        token_type_ids = batch['token_type_ids']
        targets = batch['targets']
        outputs = self.model(ids, mask, token_type_ids)
        loss = self.criterion(outputs, targets)
        self.batch_metrics = {'loss': loss}
        if self.is_train_loader:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    @torch.no_grad()
    def predict_batch(self, batch):
        batch = self._batch2device(batch, self.device)
        ids = batch['ids']
        mask = batch['mask']
        token_type_ids = batch['token_type_ids']
        outputs = self.model(ids, mask, token_type_ids)
        return outputs
