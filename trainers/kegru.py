from tqdm import tqdm
from sklearn import metrics
import torch
from datetime import datetime

from .trainer import Trainer


class KeGruTrainer(Trainer):

    def calculate_y_hat(self, data: tuple) -> tuple:
        """
        Calculate the y_hat for the given data and task

        Args:
            data (tuple): The data to calculate the y_hat for

        Returns:
            tuple: The y and y_hat
        """
        if self.task == "TaxonomyClassification":
            x, y = data
            x = x.to(self.device)
            y = y.to(self.device)
            model_input = {
                "task": self.task,
                "x": x
            }
            y_hat = self.model(model_input)
            return y, y_hat

        elif self.task == "VariantEffectPrediction":
            x1, x2, tissue, y = data
            x1 = x1.to(self.device)
            x2 = x2.to(self.device)
            tissue = tissue.to(self.device)
            y = y.to(self.device).float()
            model_input = {
                "task": self.task,
                "x1": x1,
                "x2": x2,
                "tissue": tissue
            }
            y_hat = self.model(model_input)
            return y, y_hat

        elif self.task == "PlantDeepSEA":
            x, y = data
            x = x.to(self.device)
            y = y.to(self.device)
            model_input = {
                "task": self.task,
                "x": x
            }
            y_hat = self.model(model_input)
            return y, y_hat

        else:
            raise ValueError(f"Task: {self.task} not found.")

    def calculate_predictions(self, y: torch.Tensor, y_hat: torch.Tensor) -> tuple:
        """
        Calculate the predictions for the given y and y_hat

        Args:
            y (torch.Tensor): The y
            y_hat (torch.Tensor): The y_hat

        Returns:
            tuple: The predicted and correct predictions
        """
        if self.task == "TaxonomyClassification":
            _, predicted = y_hat.max(1)
            correct_predictions = predicted.eq(y).sum().item()

        elif self.task == "VariantEffectPrediction":
            predicted = y_hat
            correct_predictions = torch.round(y_hat).eq(y).sum().item()

        elif self.task == "PlantDeepSEA":
            predicted = y_hat
            correct_predictions = (torch.round(y_hat).eq(y).sum().item() / y.size(1))
        
        else:
            raise ValueError(f"Task: {self.task} not found.")

        return predicted, correct_predictions

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

            targets.extend(y.detach().cpu().numpy())
            preds.extend(predicted.detach().cpu().numpy())

            loop.set_description(f'Epoch {current_epoch_nr}')
            loop.set_postfix(train_acc=round(correct / total, 3),
                             train_loss=round(running_loss / total, 3))

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

                targets.extend(y.detach().cpu().numpy())
                preds.extend(predicted.detach().cpu().numpy())

                loop.set_description(f'Epoch {current_epoch_nr}')
                loop.set_postfix(val_acc=round(correct / total, 3),
                                 val_loss=round(running_loss / total, 3))

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

                targets.extend(y.detach().cpu().numpy())
                preds.extend(predicted.detach().cpu().numpy())

                loop.set_description('Testing')
                loop.set_postfix(test_acc=round(correct / total, 3),
                                 test_loss=round(running_loss / total, 3))

        test_auc = metrics.roc_auc_score(targets, preds)
        test_accuracy = correct / total
        test_loss = running_loss / num_batches

        print(f"Test AUC: {test_auc}")
        print(f"Test Accuracy: {test_accuracy}")
        print(f"Test Loss: {test_loss}")

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
