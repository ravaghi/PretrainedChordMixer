from typing import Tuple
import torch

from .trainer.trainer import Trainer


class ChordMixerTrainer(Trainer):
    """ChordMixerTrainer class"""

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
        if self.task == "TaxonomyClassification":
            x, _len, _, y = batch
            x, y = x.to(self.device), y.to(self.device)

            y_hat = self.model({
                "task": self.task,
                "x": x,
                "seq_len": _len
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

        elif self.task == "PlantVariantEffectPrediction":
            x, y = batch
            x, y = x.to(self.device), y.to(self.device)

            y_hat = self.model({
                "task": self.task,
                "x": x
            })

        else:
            raise ValueError(f"Task: {self.task} not supported.")

        return y, y_hat
