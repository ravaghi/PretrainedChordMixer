from tqdm import tqdm
from sklearn import metrics
import torch
import numpy as np

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
            y = y.to(self.device)
            model_input = {
                "task": self.task,
                "x1": x1,
                "x2": x2,
                "tissue": tissue
            }
            y_hat = self.model(model_input)
            return y, y_hat

        elif self.task == "PlantDeepSEA":
            x, y, seq_len, bin = data
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

    def train(self, current_epoch_nr):
        self.model.train()

        num_batches = len(self.train_dataloader)

        running_loss = 0.0
        correct = 0
        total = 0

        train_aucs = []

        loop = tqdm(self.train_dataloader, total=num_batches)
        for batch in loop:
            y, y_hat = self.calculate_y_hat(batch)

            loss = torch.nn.functional.binary_cross_entropy(y_hat, y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            running_loss += loss.item()

            total += y.size(0)

            predictions = y_hat.cpu().detach().numpy().reshape(y_hat.shape[0])
            targets = y.cpu().numpy().reshape(y_hat.shape[0])

            cur_auc = metrics.roc_auc_score(targets, predictions)
            train_aucs.append(cur_auc)

            cur_accuracy = (predictions > 0.5) == targets
            correct += np.sum(cur_accuracy)

            loop.set_description(f'Epoch {current_epoch_nr}')
            loop.set_postfix(train_auc=round(cur_auc, 2))

        train_auc = np.mean(train_aucs)
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

        val_aucs = []

        with torch.no_grad():
            loop = tqdm(self.val_dataloader, total=num_batches)
            for batch in loop:
                y, y_hat = self.calculate_y_hat(batch)

                loss = torch.nn.functional.binary_cross_entropy(y_hat, y)

                running_loss += loss.item()
                total += y.size(0)

                predictions = y_hat.cpu().detach().numpy().reshape(y_hat.shape[0])
                targets = y.cpu().numpy().reshape(y_hat.shape[0])

                cur_auc = metrics.roc_auc_score(targets, predictions)
                val_aucs.append(cur_auc)

                cur_accuracy = (predictions > 0.5) == targets
                correct += np.sum(cur_accuracy)

                loop.set_description(f'Epoch {current_epoch_nr}')
                loop.set_postfix(val_auc=round(cur_auc, 2))

        val_auc = np.mean(val_aucs)
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

        test_aucs = []

        with torch.no_grad():
            loop = tqdm(self.test_dataloader, total=num_batches)
            for batch in loop:
                y, y_hat = self.calculate_y_hat(batch)

                loss = torch.nn.functional.binary_cross_entropy(y_hat, y)

                running_loss += loss.item()
                total += y.size(0)

                predictions = y_hat.cpu().detach().numpy().reshape(y_hat.shape[0])
                targets = y.cpu().numpy().reshape(y_hat.shape[0])

                cur_auc = metrics.roc_auc_score(targets, predictions)
                test_aucs.append(cur_auc)

                cur_accuracy = (predictions > 0.5) == targets
                correct += np.sum(cur_accuracy)

                loop.set_description(f'Testing')
                loop.set_postfix(val_auc=round(cur_auc, 2))

        test_accuracy = correct / total
        test_loss = running_loss / num_batches
        test_auc = np.mean(test_aucs)

        self.log_metrics(
            auc=test_auc,
            accuracy=test_accuracy,
            loss=test_loss,
            current_epoch_nr=-1,
            metric_type="test"
        )
