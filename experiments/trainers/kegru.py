from tqdm import tqdm
from sklearn import metrics
import torch
import numpy as np

from trainer import Trainer


class KeGruTrainer(Trainer):

    def train(self, current_epoch_nr):
        self.model.train()

        num_batches = len(self.train_dataloader)

        running_loss = 0.0
        correct = 0
        total = 0

        train_aucs = []

        loop = tqdm(self.train_dataloader, total=num_batches)
        for batch in loop:
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)

            y_hat = self.model(x)

            loss = torch.nn.functional.binary_cross_entropy(y_hat, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            total += y.size(0)

            predictions = y_hat.cpu().detach().numpy().reshape(y_hat.shape[0])
            targets = y.cpu().numpy().reshape(y_hat.shape[0])

            cur_auc = metrics.roc_auc_score(targets, predictions)
            train_aucs.append(cur_auc)

            cur_accuracy = (predictions > 0.5) == targets
            correct += np.sum(cur_accuracy)

            loop.set_description(f'Epoch {current_epoch_nr + 1}')
            loop.set_postfix(train_auc=round(cur_auc, 2))

        train_accuracy = correct / total
        train_loss = running_loss / num_batches
        train_auc = np.mean(train_aucs)

        self.log_metrics(
            auc=train_auc,
            accuracy=train_accuracy,
            loss=train_loss,
            current_epoch_nr=current_epoch_nr,
            metric_type="train"
        )

    def evaluate(self, current_epoch_nr):
        self.model.eval()

        num_batches = len(self.val_dataloader)

        running_loss = 0.0
        correct = 0
        total = 0

        val_aucs = []

        with torch.no_grad():
            loop = tqdm(self.val_dataloader, total=num_batches)
            for batch in loop:
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)

                y_hat = self.model(x)

                loss = torch.nn.functional.binary_cross_entropy(y_hat, y)

                running_loss += loss.item()
                total += y.size(0)

                predictions = y_hat.cpu().detach().numpy().reshape(y_hat.shape[0])
                targets = y.cpu().numpy().reshape(y_hat.shape[0])

                cur_auc = metrics.roc_auc_score(targets, predictions)
                val_aucs.append(cur_auc)

                cur_accuracy = (predictions > 0.5) == targets
                correct += np.sum(cur_accuracy)

                loop.set_description(f'Epoch {current_epoch_nr + 1}')
                loop.set_postfix(val_auc=round(cur_auc, 2))

        validation_accuracy = correct / total
        validation_loss = running_loss / num_batches
        val_auc = np.mean(val_aucs)

        self.log_metrics(
            auc=val_auc,
            accuracy=validation_accuracy,
            loss=validation_loss,
            current_epoch_nr=current_epoch_nr,
            metric_type="val"
        )

    def test(self):
        pass
