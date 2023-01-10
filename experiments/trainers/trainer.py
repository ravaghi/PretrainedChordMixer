from abc import ABC
import wandb


class Trainer(ABC):
    def __init__(self, model, train_dataloader, val_dataloader, test_dataloader, device, criterion, optimizer, task):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.task = task

    @staticmethod
    def log_metrics(auc: float, accuracy: float, loss: float, current_epoch_nr: int, metric_type: str) -> None:
        """
        Log metrics to wandb
        :param auc: Area under the ROC curve
        :param accuracy: Accuracy
        :param loss: Loss
        :param current_epoch_nr: Current epoch number
        :param metric_type: Type of metric (train, val, test)
        """
        if metric_type == 'train':
            wandb.log({'train_auc': auc}, step=current_epoch_nr)
            wandb.log({'train_accuracy': accuracy}, step=current_epoch_nr)
            wandb.log({'train_loss': loss}, step=current_epoch_nr)
        elif metric_type == 'val':
            wandb.log({'val_auc': auc}, step=current_epoch_nr)
            wandb.log({'val_accuracy': accuracy}, step=current_epoch_nr)
            wandb.log({'val_loss': loss}, step=current_epoch_nr)
        elif metric_type == 'test':
            wandb.run.summary['test_auc'] = auc
            wandb.run.summary['test_accuracy'] = accuracy
            wandb.run.summary['test_loss'] = loss

    def train(self, current_epoch_nr: int) -> None:
        """
        Train the model for one epoch
        :param current_epoch_nr: Current epoch number
        """
        raise NotImplementedError

    def evaluate(self, current_epoch_nr: int) -> None:
        """
        Evaluate the model for one epoch
        :param current_epoch_nr: Current epoch number
        """
        raise NotImplementedError

    def test(self) -> None:
        """
        Test the model after training
        """
        raise NotImplementedError
