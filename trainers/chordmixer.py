from sklearn import metrics
from typing import Tuple
from tqdm import tqdm
import torch

from .trainer.trainer import Trainer


class ChordMixerTrainer(Trainer):

    def calculate_y_hat(self, data: Tuple) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate y_hat for a batch depnding on the task

        Args:
            data (tuple): One batch of data

        Returns:
            tuple: y and y_hat

        Raises:
            ValueError: If the task is not supported
        """
        if self.task == "TaxonomyClassification":
            x, _len, _bin, y = data
            x, y = x.to(self.device), y.to(self.device)

            y_hat = self.model({
                "task": self.task,
                "x": x,
                "seq_len": _len
            })

        elif self.task == "HumanVariantEffectPrediction":
            x1, x2, tissue, y = data
            x1, x2, tissue, y = x1.to(self.device), x2.to(self.device), tissue.to(self.device), y.to(self.device)

            y_hat = self.model({
                "task": self.task,
                "x1": x1,
                "x2": x2,
                "tissue": tissue
            })

        elif self.task == "PlantVariantEffectPrediction":
            x, y = data
            x, y = x.to(self.device), y.to(self.device)

            y_hat = self.model({
                "task": self.task,
                "x": x
            })

        else:
            raise ValueError(f"Task: {self.task} not supported.")

        return y, y_hat

    def calculate_predictions(self, y: torch.Tensor, y_hat: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Calculate predictions and the number of correct predictions

        Args:
            y (torch.Tensor): The y
            y_hat (torch.Tensor): The y_hat

        Returns:
            tuple: Predictions and the number of correct predictions
        """
        if self.task == "TaxonomyClassification":
            _, predicted = y_hat.max(1)
            correct_predictions = predicted.eq(y).sum().item()

        elif self.task == "HumanVariantEffectPrediction":
            predicted = y_hat
            correct_predictions = torch.round(y_hat).eq(y).sum().item()

        elif self.task == "PlantVariantEffectPrediction":
            predicted = y_hat
            correct_predictions = (torch.round(y_hat).eq(y).sum().item() / y.size(1))

        else:
            raise ValueError(f"Task: {self.task} not found.")

        return predicted, correct_predictions

    def train(self, current_epoch_nr: int) -> None:
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

            targets.extend(y.detach().cpu().numpy())
            preds.extend(predicted.detach().cpu().numpy())

            loop.set_description(f'Epoch {current_epoch_nr}')
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

    def evaluate(self, current_epoch_nr: int) -> None:
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

                targets.extend(y.detach().cpu().numpy())
                preds.extend(predicted.detach().cpu().numpy())

                loop.set_description(f'Epoch {current_epoch_nr}')
                loop.set_postfix(val_acc=round(correct / total, 3),
                                 val_loss=round(running_loss / total, 3))

        validation_auc = metrics.roc_auc_score(targets, preds)
        validation_accuracy = correct / total
        validation_loss = running_loss / num_batches

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

                targets.extend(y.detach().cpu().numpy())
                preds.extend(predicted.detach().cpu().numpy())

                loop.set_description('Testing')
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
