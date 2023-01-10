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

            if self.task == "taxonomy_classification":
                x, y, seq_len, bin = batch
                x = x.to(self.device)
                y = y.to(self.device)
                model_input = {
                    "task": "taxonomy_classification",
                    "x": x,
                    "seq_len": seq_len
                }
                y_hat = self.model(model_input)
            
            elif self.task == "variant_effect_prediction":
                x1, x2, tissue, y = batch
                x1 = x1.to(self.device)
                x2 = x2.to(self.device)
                tissue = tissue.to(self.device)
                y = y.to(self.device).float()
                model_input = {
                    "task": "variant_effect_prediction",
                    "x1": x1,
                    "x2": x2,
                    "tissue": tissue
                }
                y_hat = self.model(model_input)

            elif self.task == "plantdeepsea":
                x, y, seq_len, bin = batch
                x = x.to(self.device)
                y = y.to(self.device)
                model_input = {
                    "task": "plantdeepsea",
                    "x": x
                }
                y_hat = self.model(model_input)

            loss = self.criterion(y_hat, y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            running_loss += loss.item()

            if self.task == "taxonomy_classification":
                _, predicted = y_hat.max(1)
                correct += predicted.eq(y).sum().item()

            elif self.task == "variant_effect_prediction":
                predicted = y_hat
                correct += torch.round(y_hat).eq(y).sum().item()

            elif self.task == "plantdeepsea":
                predicted = y_hat
                correct += (torch.round(y_hat).eq(y).sum().item() / y.size(1))            

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
                if self.task == "taxonomy_classification":
                    x, y, seq_len, bin = batch
                    x = x.to(self.device)
                    y = y.to(self.device)
                    model_input = {
                        "task": "taxonomy_classification",
                        "x": x,
                        "seq_len": seq_len
                    }
                    y_hat = self.model(model_input)
            
                elif self.task == "variant_effect_prediction":
                    x1, x2, tissue, y = batch
                    x1 = x1.to(self.device)
                    x2 = x2.to(self.device)
                    tissue = tissue.to(self.device)
                    y = y.to(self.device).float()
                    model_input = {
                        "task": "variant_effect_prediction",
                        "x1": x1,
                        "x2": x2,
                        "tissue": tissue
                    }
                    y_hat = self.model(model_input)

                elif self.task == "plantdeepsea":
                    x, y, seq_len, bin = batch
                    x = x.to(self.device)
                    y = y.to(self.device)
                    model_input = {
                        "task": "plantdeepsea",
                        "x": x
                    }
                    y_hat = self.model(model_input)

                loss = self.criterion(y_hat, y)

                running_loss += loss.item()

                if self.task == "taxonomy_classification":
                    _, predicted = y_hat.max(1)
                    correct += predicted.eq(y).sum().item()

                elif self.task == "variant_effect_prediction":
                    predicted = y_hat
                    correct += torch.round(y_hat).eq(y).sum().item()

                elif self.task == "plantdeepsea":
                    predicted = y_hat
                    correct += (torch.round(y_hat).eq(y).sum().item() / y.size(1))

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
                if self.task == "taxonomy_classification":
                    x, y, seq_len, bin = batch
                    x = x.to(self.device)
                    y = y.to(self.device)
                    model_input = {
                        "task": "taxonomy_classification",
                        "x": x,
                        "seq_len": seq_len
                    }
                    y_hat = self.model(model_input)
            
                elif self.task == "variant_effect_prediction":
                    x1, x2, tissue, y = batch
                    x1 = x1.to(self.device)
                    x2 = x2.to(self.device)
                    tissue = tissue.to(self.device)
                    y = y.to(self.device).float()
                    model_input = {
                        "task": "variant_effect_prediction",
                        "x1": x1,
                        "x2": x2,
                        "tissue": tissue
                    }
                    y_hat = self.model(model_input)

                elif self.task == "plantdeepsea":
                    x, y, seq_len, bin = batch
                    x = x.to(self.device)
                    y = y.to(self.device)
                    model_input = {
                        "task": "plantdeepsea",
                        "x": x
                    }
                    y_hat = self.model(model_input)

                loss = self.criterion(y_hat, y)

                running_loss += loss.item()

                if self.task == "taxonomy_classification":
                    _, predicted = y_hat.max(1)
                    correct += predicted.eq(y).sum().item()

                elif self.task == "variant_effect_prediction":
                    predicted = y_hat
                    correct += torch.round(y_hat).eq(y).sum().item()

                elif self.task == "plantdeepsea":
                    predicted = y_hat
                    correct += (torch.round(y_hat).eq(y).sum().item() / y.size(1))

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

