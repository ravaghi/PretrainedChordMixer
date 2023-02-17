from abc import ABC
import wandb
import torch
import os


class Trainer(ABC):
    def __init__(self, device, model, criterion, optimizer, task, train_dataloader, val_dataloader, test_dataloader, log_to_wandb, save_dir):
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

    def save_model(self, model, name) -> None:
        """
        Save the model and log to wandb

        Args:
            model (torch.nn.Module): Model to save
            name (str): Name of the model

        Returns:
            None
        """
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        model_path = os.path.join(self.save_dir, name)

        # Save model to disk
        torch.save(model.state_dict(), model_path)

        if self.log_to_wandb:
            # Save model to wandb
            artifact = wandb.Artifact(name, type='model')
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
            metric_type (str): Type of metric

        Returns:
            None
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

        Args:
            current_epoch_nr (int): Current epoch number

        Returns:
            None
        """
        raise NotImplementedError

    def evaluate(self, current_epoch_nr: int) -> None:
        """
        Evaluate the model for one epoch

        Args:
            current_epoch_nr (int): Current epoch number

        Returns:
            None
        """
        raise NotImplementedError

    def test(self) -> None:
        """
        Test the model after training

        Returns:
            None
        """
        raise NotImplementedError
