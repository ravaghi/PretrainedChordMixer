from datetime import datetime
from sklearn import metrics
from tqdm import tqdm
import torch

from .trainer.trainer import Trainer


class XFormerTrainer(Trainer):

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
            y, y_hat = self.calculate_y_hat(batch)

            loss = self.criterion(y_hat, y)
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            running_loss += loss.item()

            predicted, correct_predictions = self.calculate_predictions(y, y_hat)
            correct += correct_predictions
            total += y.size(0)

            targets.extend(y.detach().cpu().numpy().flatten())
            preds.extend(predicted.detach().cpu().numpy().flatten())

            loop.set_description(f'Epoch {current_epoch_nr}')
            loop.set_postfix(train_acc=round(correct / total, 2),
                             train_loss=round(running_loss / total, 2))

        train_auc = metrics.roc_auc_score(targets, preds)
        train_accuracy = correct / total
        train_loss = running_loss / num_batches

        if self.log_to_wandb:
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
                y, y_hat = self.calculate_y_hat(batch)

                loss = self.criterion(y_hat, y)

                running_loss += loss.item()

                predicted, correct_predictions = self.calculate_predictions(y, y_hat)
                correct += correct_predictions
                total += y.size(0)

                targets.extend(y.detach().cpu().numpy().flatten())
                preds.extend(predicted.detach().cpu().numpy().flatten())

                loop.set_description(f'Epoch {current_epoch_nr}')
                loop.set_postfix(val_acc=round(correct / total, 2),
                                 val_loss=round(running_loss / total, 2))

        validation_auc = metrics.roc_auc_score(targets, preds)
        validation_accuracy = correct / total
        validation_loss = running_loss / num_batches

        if self.log_to_wandb:
            self.log_metrics(
                auc=validation_auc,
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
                y, y_hat = self.calculate_y_hat(batch)

                loss = self.criterion(y_hat, y)

                running_loss += loss.item()

                predicted, correct_predictions = self.calculate_predictions(y, y_hat)
                correct += correct_predictions
                total += y.size(0)

                targets.extend(y.detach().cpu().numpy().flatten())
                preds.extend(predicted.detach().cpu().numpy().flatten())

                loop.set_description(f'Testing')
                loop.set_postfix(val_acc=round(correct / total, 2),
                                 val_loss=round(running_loss / total, 2))

        test_auc = metrics.roc_auc_score(targets, preds)
        test_accuracy = correct / total
        test_loss = running_loss / num_batches

        if self.log_to_wandb:
            self.log_metrics(
                auc=test_auc,
                accuracy=test_accuracy,
                loss=test_loss,
                current_epoch_nr=-1,
                metric_type="test"
            )

        current_datetime = datetime.now().strftime("%d%b%Y_%H%M%S")
        self.save_model(self.model, f"{current_datetime}-AUC-{test_auc:.4f}.pt")
