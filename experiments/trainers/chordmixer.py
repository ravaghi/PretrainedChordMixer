from tqdm import tqdm
from sklearn import metrics
import torch

from .trainer import Trainer


class ChordMixerTrainer(Trainer):
    def train(self, current_epoch_nr):
        self.model.train()

        num_batches = len(self.train_dataloader)

        running_loss = 0.0
        correct = 0
        total = 0

        preds = []
        targets = []

        loop = tqdm(self.train_dataloader, total=num_batches)
        for batch in loop:
            x, y, seq_len, _bin = batch
            x = x.to(self.device)
            y = y.to(self.device)

            if self.model.variable_length:
                y_hat = self.model(x, seq_len)
            else:
                y_hat = self.model(x)

            loss = self.criterion(y_hat, y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            running_loss += loss.item()

            if self.model.n_class > 2:
                predicted = y_hat
                correct += (torch.round(y_hat).eq(y).sum().item() / y.size(1))
            else:
                _, predicted = y_hat.max(1)
                correct += predicted.eq(y).sum().item()

            total += y.size(0)

            targets.extend(y.detach().cpu().numpy())
            preds.extend(predicted.detach().cpu().numpy())

            loop.set_description(f'Epoch {current_epoch_nr + 1}')
            loop.set_postfix(train_acc=round(correct / total, 3),
                             train_loss=round(running_loss / total, 3))

        train_auc = metrics.roc_auc_score(targets, preds)
        train_accuracy = correct / total
        train_loss = running_loss / num_batches

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

        preds = []
        targets = []

        with torch.no_grad():
            loop = tqdm(self.val_dataloader, total=num_batches)
            for batch in loop:
                x, y, seq_len, _bin = batch
                x = x.to(self.device)
                y = y.to(self.device)

                if self.model.variable_length:
                    y_hat = self.model(x, seq_len)
                else:
                    y_hat = self.model(x)

                loss = self.criterion(y_hat, y)

                running_loss += loss.item()

                if self.model.n_class > 2:
                    predicted = y_hat
                    correct += (torch.round(y_hat).eq(y).sum().item() / y.size(1))
                else:
                    _, predicted = y_hat.max(1)
                    correct += predicted.eq(y).sum().item()

                total += y.size(0)
                
                targets.extend(y.detach().cpu().numpy())
                preds.extend(predicted.detach().cpu().numpy())

                loop.set_description(f'Epoch {current_epoch_nr + 1}')
                loop.set_postfix(val_acc=round(correct / total, 3),
                                 val_loss=round(running_loss / total, 3))

        val_auc = metrics.roc_auc_score(targets, preds)
        validation_accuracy = correct / total
        validation_loss = running_loss / num_batches

        self.log_metrics(
            auc=val_auc,
            accuracy=validation_accuracy,
            loss=validation_loss,
            current_epoch_nr=current_epoch_nr,
            metric_type="val"
        )

    def test(self):
        self.model.eval()

        num_batches = len(self.test_dataloader)

        running_loss = 0.0
        correct = 0
        total = 0

        preds = []
        targets = []

        with torch.no_grad():
            loop = tqdm(self.test_dataloader, total=num_batches)
            for batch in loop:
                x, y, seq_len, _bin = batch
                x = x.to(self.device)
                y = y.to(self.device)

                if self.model.variable_length:
                    y_hat = self.model(x, seq_len)
                else:
                    y_hat = self.model(x)

                loss = self.criterion(y_hat, y)

                running_loss += loss.item()

                if self.model.n_class > 2:
                    predicted = y_hat
                    correct += (torch.round(y_hat).eq(y).sum().item() / y.size(1))
                else:
                    _, predicted = y_hat.max(1)
                    correct += predicted.eq(y).sum().item()

                total += y.size(0)

                targets.extend(y.detach().cpu().numpy())
                preds.extend(predicted.detach().cpu().numpy())

                loop.set_description(f'Testing')
                loop.set_postfix(test_acc=round(correct / total, 3),
                                 test_loss=round(running_loss / total, 3))

        test_auc = metrics.roc_auc_score(targets, preds)
        test_accuracy = correct / total
        test_loss = running_loss / num_batches

        self.log_metrics(
            auc=test_auc,
            accuracy=test_accuracy,
            loss=test_loss,
            current_epoch_nr=-1,
            metric_type="test"
        )

