from datetime import datetime
from sklearn import metrics
from torch import Tensor
from typing import Tuple
from tqdm import tqdm
import numpy as np
import torch
import wandb

from .trainer.trainer import Trainer


class PretrainedChordMixerTrainer(Trainer):
    """Trainer for pretrained model."""

    def _calculate_y_hat(self, batch: Tuple[Tensor, Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Calculates y_hat for a batch.

        Args:
            batch Tuple[Tensor, Tensor, Tensor, Tensor]: A batch of data, containing the sequence ids, masks, labels and lengths.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Tuple containing y, y_hat and masks
        """
        sequence_ids, masks, labels, lengths = batch

        x = sequence_ids.to(self.device)
        y = labels.to(self.device)
        masks = masks.to(self.device)

        y_hat = self.model(x, lengths)

        return y, y_hat, masks

    @staticmethod
    def _calcualte_predictions(y: Tensor, y_hat: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        """
        returns predictions and labels, only at the masked positions.

        Args:
            y (Tensor): Labels.
            y_hat (Tensor): Model predictions.
            mask (Tensor): Mask.

        Returns:
            Tuple[Tensor, Tensor]: Tuple containing the labels and predictions.
        """
        y = y.masked_select(mask)
        y_hat = y_hat.argmax(dim=-1).masked_select(mask)
        return y, y_hat

    @staticmethod
    def _calculate_auc(y: Tensor, y_hat: Tensor, mask: Tensor) -> float:
        """
        Calculates the ROC-AUC score for a batch.

        Args:
            y (Tensor): Labels.
            y_hat (Tensor): Model predictions.
            mask (Tensor): Mask.

        Returns:
            float: ROC-AUC score for a batch.
        """
        try:
            target = y.masked_select(mask).detach().cpu().numpy()
            # need to take the inverse since metrics.roc_auc_score expects probabilities, but we have logsoftmax values
            prediction = torch.exp(y_hat[mask == True]).detach().cpu().numpy()
            return metrics.roc_auc_score(target, prediction, multi_class='ovo')
        except ValueError:
            return 0.5

    @staticmethod
    def _log_metrics(auc: float, accuracy: float, loss: float, metric_type: str) -> None:
        """
        Logs metrics to wandb.

        Args:
            auc (float): ROC-AUC score.
            accuracy (float): Accuracy.
            loss (float): Loss.
            metric_type (str): Type of metric, either train, val or test.

        Returns:
            None
        """
        if metric_type == 'train':
            wandb.log({
                'train_auc': auc,
                'train_accuracy': accuracy,
                'train_loss': loss
            })
        elif metric_type == 'val':
            wandb.log({
                'val_auc': auc,
                'val_accuracy': accuracy,
                'val_loss': loss
            })
        elif metric_type == 'test':
            wandb.run.summary['test_auc'] = auc
            wandb.run.summary['test_accuracy'] = accuracy
            wandb.run.summary['test_loss'] = loss

    def train(self, current_epoch_nr: int) -> None:
        """
        Trains the model for one epoch.

        Args:
            current_epoch_nr (int): Current epoch number.

        Returns:
            None
        """
        self.model.train()

        num_batches = len(self.train_dataloader)
        log_interval = num_batches // 100

        running_loss = 0.0

        accuracies = []
        aucs = []

        loop = tqdm(self.train_dataloader, total=num_batches)
        for idx, batch in enumerate(loop):
            self.optimizer.zero_grad()

            y, y_hat, masks = self._calculate_y_hat(batch)

            # Filling unmasked positions with 0
            token_mask = ~masks.unsqueeze(-1).expand_as(y_hat)
            y_hat = y_hat.masked_fill(token_mask, 0)

            loss = self.criterion(y_hat.transpose(1, 2), y)
            loss.backward()
            self.optimizer.step()

            current_loss = loss.item()
            running_loss += current_loss

            target, prediction = self._calcualte_predictions(y, y_hat, masks)

            current_accuracy = metrics.accuracy_score(
                target.detach().cpu().numpy(),
                prediction.detach().cpu().numpy()
            )
            current_auc = self._calculate_auc(y, y_hat, masks)

            accuracies.append(current_accuracy)
            aucs.append(current_auc)

            loop.set_description(f'Epoch {current_epoch_nr}')
            loop.set_postfix(train_acc=round(current_accuracy, 5),
                             train_loss=round(current_loss, 5))

            if self.scheduler is not None:
                self.scheduler.step()

            if (idx + 1) % log_interval == 0 and self.log_to_wandb:
                train_auc = float(np.mean(aucs))
                train_accuracy = float(np.mean(accuracies))
                train_loss = running_loss / log_interval
                self._log_metrics(train_auc, train_accuracy, train_loss, 'train')
                running_loss = 0.0
                accuracies = []
                aucs = []

    def evaluate(self, current_epoch_nr) -> None:
        """
        Evaluates the model on the validation set.

        Args:
            current_epoch_nr (int): Current epoch number.

        Returns:
            None
        """
        self.model.eval()

        num_batches = len(self.val_dataloader)

        accuracies = []
        aucs = []

        with torch.no_grad():
            loop = tqdm(self.val_dataloader, total=num_batches)
            for batch in loop:
                y, y_hat, masks = self._calculate_y_hat(batch)

                # Filling unmasked positions with 0
                token_mask = ~masks.unsqueeze(-1).expand_as(y_hat)
                y_hat = y_hat.masked_fill(token_mask, 0)

                loss = self.criterion(y_hat.transpose(1, 2), y)

                current_loss = loss.item()

                target, prediction = self._calcualte_predictions(y, y_hat, masks)

                current_accuracy = metrics.accuracy_score(
                    target.detach().cpu().numpy(),
                    prediction.detach().cpu().numpy()
                )
                current_auc = self._calculate_auc(y, y_hat, masks)

                accuracies.append(current_accuracy)
                aucs.append(current_auc)

                loop.set_description(f'Epoch {current_epoch_nr}')
                loop.set_postfix(val_acc=round(current_accuracy, 5),
                                 val_loss=round(current_loss, 5))

        val_auc = float(np.mean(aucs))
        val_accuracy = float(np.mean(accuracies))

        if self.log_to_wandb:
            current_datetime = datetime.now().strftime("%d%b%Y_%H%M%S")
            model_name = f"{current_datetime}-ValAuc-{val_auc:.4f}-ValAcc-{val_accuracy:.4f}"
            self.save_model(model=self.model, name=model_name)

    def test(self) -> None:
        """
        Tests the model on the test set.

        Args:
            None

        Returns:
            None
        """
        self.model.eval()

        num_batches = len(self.test_dataloader)

        running_loss = 0.0

        aucs = []
        accuracies = []

        with torch.no_grad():
            loop = tqdm(self.test_dataloader, total=num_batches)
            for batch in loop:
                y, y_hat, masks = self._calculate_y_hat(batch)

                # Filling unmasked positions with 0
                token_mask = ~masks.unsqueeze(-1).expand_as(y_hat)
                y_hat = y_hat.masked_fill(token_mask, 0)

                loss = self.criterion(y_hat.transpose(1, 2), y)

                current_loss = loss.item()
                running_loss += current_loss

                target, prediction = self._calcualte_predictions(y, y_hat, masks)

                current_accuracy = metrics.accuracy_score(
                    target.detach().cpu().numpy(),
                    prediction.detach().cpu().numpy()
                )
                current_auc = self._calculate_auc(y, y_hat, masks)

                accuracies.append(current_accuracy)
                aucs.append(current_auc)

                loop.set_description(f'Testing')
                loop.set_postfix(test_acc=round(current_accuracy, 5),
                                 test_loss=round(current_loss, 5))

        test_auc = float(np.mean(aucs))
        test_accuracy = float(np.mean(accuracies))
        test_loss = running_loss / num_batches

        if self.log_to_wandb:
            self._log_metrics(test_auc, test_accuracy, test_loss, 'test')
            current_datetime = datetime.now().strftime("%d%b%Y_%H%M%S")
            model_name = f"{current_datetime}-TestAuc-{test_auc:.4f}-TestAcc-{test_accuracy:.4f}"
            self.save_model(model=self.model, name=model_name)
