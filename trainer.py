from tqdm import tqdm
from sklearn import metrics
import torch
import wandb
from typing import Tuple
import numpy as np
from torch import Tensor


class PretrainedChordMixerTrainer:
    def __init__(self,
                 model,
                 train_dataloader,
                 test_dataloader,
                 device,
                 criterion,
                 optimizer,
                 log_interval,
                 log_to_wandb):
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.log_interval = log_interval
        self.log_to_wandb = log_to_wandb

    def _calculate_y_hat(self, batch: Tuple[Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Calculates y_hat for a batch

        Args:
            batch Tuple[Tensor, Tensor, Tensor]: A batch of data

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Tuple containing y, y_hat and masks
        """
        sequence_ids, masks, labels = batch

        x: Tensor = sequence_ids.to(self.device)
        y: Tensor = labels.to(self.device)
        masks: Tensor = masks.to(self.device)

        y_hat: Tensor = self.model(x)

        return y, y_hat, masks

    @staticmethod
    def _calcualte_predictions(y: Tensor, y_hat: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Selects the predictions and labels based on the mask positions

        Args:
            y (Tensor): The labels
            y_hat (Tensor): The predictions
            mask (Tensor): The mask

        Returns:
            Tuple[Tensor, Tensor]: Tuple containing the labels and predictions
        """
        y = y.masked_select(mask)
        y_hat = y_hat.argmax(dim=-1).masked_select(mask)
        return y, y_hat

    @staticmethod
    def _calculate_auc(y: Tensor, y_hat: Tensor, mask: Tensor) -> float:
        """
        Calculates the AUC for a batch

        Args:
            y (Tensor): The labels
            y_hat (Tensor): The predictions
            mask (Tensor): The mask

        Returns:
            float: The AUC
        """
        try:
            target = y.masked_select(mask).detach().cpu().numpy()
            # need to take the inverse since metrics.roc_auc_score expects probabilities, but we have logsoftmax values
            prediction = torch.exp(y_hat[mask == True]).detach().cpu().numpy()
            return metrics.roc_auc_score(target, prediction, multi_class='ovr')
        except ValueError:
            return 0.0

    @staticmethod
    def log_metrics(auc: float, accuracy: float, loss: float, metric_type: str) -> None:
        """
        Log metrics to wandb

        Args:
            auc (float): Area under the curve
            accuracy (float): Accuracy
            loss (float): Loss
            metric_type (str): Type of metric

        Returns:
            None
        """
        if metric_type == 'train':
            wandb.log({'train_auc': auc})
            wandb.log({'train_accuracy': accuracy})
            wandb.log({'train_loss': loss})
        elif metric_type == 'test':
            wandb.run.summary['test_auc'] = auc
            wandb.run.summary['test_accuracy'] = accuracy
            wandb.run.summary['test_loss'] = loss

    def train(self, current_epoch_nr: int) -> None:
        """
        Train the model for one epoch

        Args:
            current_epoch_nr (int): The current epoch number

        Returns:
            None
        """
        self.model.train()

        num_batches = len(self.train_dataloader)

        running_loss = 0.0
        total = 0

        targets = []
        predictions = []

        loop = tqdm(self.train_dataloader, total=num_batches)
        for idx, batch in enumerate(loop):
            self.optimizer.zero_grad()

            y, y_hat, masks = self._calculate_y_hat(batch)

            # Filling unmasked positions with -1e3
            token_mask = ~masks.unsqueeze(-1).expand_as(y_hat)
            y_hat = y_hat.masked_fill(token_mask, -1e3)

            loss = self.criterion(y_hat.transpose(1, 2), y)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            total += y.size(0)
            current_loss = running_loss / total

            target, prediction = self._calcualte_predictions(y, y_hat, masks)

            targets.extend(target.detach().cpu().numpy())
            predictions.extend(prediction.detach().cpu().numpy())

            current_accuracy = metrics.accuracy_score(targets, predictions)
            current_auc = self._calculate_auc(y, y_hat, masks)

            loop.set_description(f'Epoch {current_epoch_nr}')
            loop.set_postfix(train_loss=round(current_loss, 5),
                             train_accuracy=round(current_accuracy, 5))

            if (idx + 1) % self.log_interval == 0 and self.log_to_wandb:
                targets = []
                predictions = []
                self.log_metrics(current_auc, current_accuracy, current_loss, 'train')

    def test(self) -> None:
        """
        Test the model

        Returns:
            None
        """
        self.model.eval()

        num_batches = len(self.test_dataloader)

        running_loss = 0.0
        total = 0

        targets = []
        predictions = []

        aucs = []

        with torch.no_grad():
            loop = tqdm(self.test_dataloader, total=num_batches)
            for batch in loop:
                y, y_hat, masks = self._calculate_y_hat(batch)

                # Filling unmasked positions with -1e3
                token_mask = ~masks.unsqueeze(-1).expand_as(y_hat)
                y_hat = y_hat.masked_fill(token_mask, -1e3)

                loss = self.criterion(y_hat.transpose(1, 2), y)

                running_loss += loss.item()
                total += y.size(0)
                current_loss = running_loss / total

                target, prediction = self._calcualte_predictions(y, y_hat, masks)

                targets.extend(target.detach().cpu().numpy())
                predictions.extend(prediction.detach().cpu().numpy())

                current_accuracy = metrics.accuracy_score(targets, predictions)
                current_auc = self._calculate_auc(y, y_hat, masks)

                aucs.append(current_auc)

                loop.set_description(f'Test')
                loop.set_postfix(test_loss=round(current_loss, 5),
                                 test_accuracy=round(current_accuracy, 5))

        test_loss = running_loss / total
        test_accuracy = metrics.accuracy_score(targets, predictions)
        test_auc = float(np.mean(aucs))

        if self.log_to_wandb:
            self.log_metrics(test_auc, test_accuracy, test_loss, 'test')
        else:
            print(f'Test loss: {test_loss}, Test accuracy: {test_accuracy}, Test AUC: {test_auc}')
