from sklearn.metrics import roc_auc_score
from datetime import datetime
from typing import Tuple
from tqdm import tqdm
from abc import ABC
import wandb
import torch
import os


class Trainer(ABC):
    """Trainer class"""

    def __init__(self,
                 device,
                 model,
                 criterion,
                 optimizer,
                 task,
                 train_dataloader,
                 val_dataloader,
                 test_dataloader,
                 log_to_wandb,
                 save_dir,
                 scheduler=None
                 ):
        self.device = device
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.task = task
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.log_to_wandb = log_to_wandb
        self.save_dir = save_dir
        self.scheduler = scheduler

    def save_model(self, model: torch.nn.Module, name: str) -> None:
        """
        Save the model to disk, and log it to wandb as an artifact of the run

        Args:
            model (torch.nn.Module): Model to save
            name (str): Name of the model

        Returns:
            None
        """

        def _parse_model_name():
            dir_list = self.save_dir.split("/")[7:]
            model_name = ""
            for index, dir in enumerate(dir_list):
                if index == (len(dir_list) - 1):
                    model_name += dir
                else:
                    model_name += dir + "-"
            return model_name

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        model_path = os.path.join(self.save_dir, name + ".pt")

        # Save model to disk
        torch.save(model.state_dict(), model_path)

        if self.log_to_wandb:
            # Save model to wandb
            artifact = wandb.Artifact(_parse_model_name(), type='model')
            artifact.add_file(model_path)
            wandb.run.log_artifact(artifact)

    @staticmethod
    def log_metrics(auc: float, accuracy: float, loss: float, current_epoch_nr: int, metric_type: str) -> None:
        """
        Log metrics to wandb

        Args:
            auc (float): Area under the curve
            accuracy (float): Accuracy
            loss (float): Loss
            current_epoch_nr (int): Current epoch number
            metric_type (str): Type of metric, train, val or test

        Returns:
            None
        """
        if metric_type == 'train':
            wandb.log({
                'train_auc': auc,
                'train_accuracy': accuracy,
                'train_loss': loss
            }, step=current_epoch_nr)
        elif metric_type == 'val':
            wandb.log({
                'val_auc': auc,
                'val_accuracy': accuracy,
                'val_loss': loss
            }, step=current_epoch_nr)
        elif metric_type == 'test':
            wandb.run.summary['test_auc'] = auc
            wandb.run.summary['test_accuracy'] = accuracy
            wandb.run.summary['test_loss'] = loss

    def calculate_y_hat(self, batch: Tuple) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate y_hat for a batch depnding on the task

        Args:
            batch (tuple): One batch of data

        Returns:
            tuple: y and y_hat

        Raises:
            ValueError: If the task is not supported
        """
        if self.task in ["TaxonomyClassification", "PlantVariantEffectPrediction"]:
            x, y = batch
            x, y = x.to(self.device), y.to(self.device)
            y_hat = self.model({
                "task": self.task,
                "x": x
            })

        elif self.task == "HumanVariantEffectPrediction":
            x1, x2, tissue, y = batch
            x1, x2, tissue, y = x1.to(self.device), x2.to(self.device), tissue.to(self.device), y.to(self.device)

            y_hat = self.model({
                "task": self.task,
                "x1": x1,
                "x2": x2,
                "tissue": tissue
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
        if self.task in ["TaxonomyClassification", "HumanVariantEffectPrediction"]:
            predicted = torch.round(torch.sigmoid(y_hat))
            correct_predictions = predicted.eq(y).sum().item()

        elif self.task == "PlantVariantEffectPrediction":
            predicted = torch.round(torch.sigmoid(y_hat))
            correct_predictions = predicted.eq(y).sum().item() // y.size(1)

        else:
            raise ValueError(f"Task: {self.task} not found.")

        return predicted, correct_predictions

    def train(self, current_epoch_nr: int) -> None:
        """
        Train the model for one epoch

        Args:
            current_epoch_nr (int): Current epoch number

        Returns:
            None
        """
        self.model.train()

        num_batches = len(self.train_dataloader)

        running_loss = 0.0
        correct = 0
        total = 0

        targets = []
        preds = []

        loop = tqdm(self.train_dataloader, total=num_batches)
        for batch in loop:
            self.optimizer.zero_grad()

            y, y_hat = self.calculate_y_hat(batch=batch)

            loss = self.criterion(y_hat, y)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            predicted, correct_predictions = self.calculate_predictions(y=y, y_hat=y_hat)

            correct += correct_predictions
            total += y.size(0)

            targets.extend(y.cpu().detach().numpy())
            preds.extend(predicted.cpu().detach().numpy())

            loop.set_description(f"Epoch {current_epoch_nr}")
            loop.set_postfix(train_acc=round(correct / total, 5),
                             train_loss=round(running_loss / total, 5))

        train_auc = roc_auc_score(y_true=targets, y_score=preds)
        train_accuracy = correct / total
        train_loss = running_loss / num_batches

        if self.log_to_wandb:
            self.log_metrics(
                auc=train_auc,
                accuracy=train_accuracy,
                loss=train_loss,
                current_epoch_nr=current_epoch_nr,
                metric_type='train'
            )

    def evaluate(self, current_epoch_nr: int) -> None:
        """
        Evaluate the model for one epoch

        Args:
            current_epoch_nr (int): Current epoch number

        Returns:
            None
        """
        self.model.eval()

        num_batches = len(self.val_dataloader)

        running_loss = 0.0
        correct = 0
        total = 0

        targets = []
        preds = []

        with torch.no_grad():
            loop = tqdm(self.val_dataloader, total=num_batches)
            for batch in loop:
                y, y_hat = self.calculate_y_hat(batch=batch)

                loss = self.criterion(y_hat, y)

                running_loss += loss.item()

                predicted, correct_predictions = self.calculate_predictions(y=y, y_hat=y_hat)

                correct += correct_predictions
                total += y.size(0)

                targets.extend(y.detach().cpu().numpy())
                preds.extend(predicted.detach().cpu().numpy())

                loop.set_description(f'Epoch {current_epoch_nr}')
                loop.set_postfix(val_acc=round(correct / total, 5),
                                 val_loss=round(running_loss / total, 5))

        validation_auc = roc_auc_score(y_true=targets, y_score=preds)
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

    def test(self) -> None:
        """
        Test the model after training

        Returns:
            None
        """
        self.model.eval()

        num_batches = len(self.test_dataloader)

        running_loss = 0.0
        correct = 0
        total = 0

        targets = []
        preds = []

        with torch.no_grad():
            loop = tqdm(self.test_dataloader, total=num_batches)
            for batch in loop:
                y, y_hat = self.calculate_y_hat(batch=batch)

                loss = self.criterion(y_hat, y)

                running_loss += loss.item()

                predicted, correct_predictions = self.calculate_predictions(y=y, y_hat=y_hat)

                correct += correct_predictions
                total += y.size(0)

                targets.extend(y.detach().cpu().numpy())
                preds.extend(predicted.detach().cpu().numpy())

                loop.set_description('Testing')
                loop.set_postfix(test_acc=round(correct / total, 5),
                                 test_loss=round(running_loss / total, 5))

        test_auc = roc_auc_score(y_true=targets, y_score=preds)
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
        model_name = f"{current_datetime}-AUC-{test_auc:.4f}"
        self.save_model(model=self.model, name=model_name)
