from tqdm import tqdm
from sklearn import metrics
import torch
import wandb


class PretrainedChordMixerTrainer:
    def __init__(self, model, train_dataloader, val_dataloader, test_dataloader, device, criterion, optimizer):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer

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

        loop = tqdm(self.train_dataloader, total=num_batches)
        for idx, batch in enumerate(loop):
            sequence_ids, masks, labels = batch

            x = sequence_ids.to(self.device)
            y = labels.to(self.device)
            masks = masks.to(self.device)
            
            y_hat = self.model(x)

            y = y[masks == 1]
            y_hat = y_hat[masks == 1]

            loss = self.criterion(y_hat, y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            running_loss += loss.item()

            total += y.size(0)

            loop.set_description(f'Epoch {current_epoch_nr}')
            loop.set_postfix(train_loss=round(running_loss / total, 5))

            if idx % 100 == 0:
                wandb.log({'train_loss': running_loss / total})

    def evaluate(self, current_epoch_nr: int) -> None:
        """
        Evaluate the model
        
        Args:
            current_epoch_nr (int): The current epoch number

        Returns:
            None
        """
        self.model.eval()

        num_batches = len(self.val_dataloader)

        running_loss = 0.0
        total = 0

        with torch.no_grad():
            loop = tqdm(self.val_dataloader, total=num_batches)
            for idx, batch in enumerate(loop):
                sequence_ids, masks, labels = batch

                x = sequence_ids.to(self.device)
                y = labels.to(self.device)
                masks = masks.to(self.device)

                y_hat = self.model(x)

                y = y[masks == 1]
                y_hat = y_hat[masks == 1]

                loss = self.criterion(y_hat, y)

                running_loss += loss.item()

                total += y.size(0)

                loop.set_description(f'Epoch {current_epoch_nr}')
                loop.set_postfix(val_loss=round(running_loss / total, 5))

                if idx % 100 == 0:
                    wandb.log({'val_loss': running_loss / total})


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

        with torch.no_grad():
            loop = tqdm(self.test_dataloader, total=num_batches)
            for idx, batch in enumerate(loop):
                sequence_ids, masks, labels = batch

                x = sequence_ids.to(self.device)
                y = labels.to(self.device)
                masks = masks.to(self.device)

                y_hat = self.model(x)

                y = y[masks == 1]
                y_hat = y_hat[masks == 1]

                loss = self.criterion(y_hat, y)

                running_loss += loss.item()

                total += y.size(0)

                loop.set_description(f'Test')
                loop.set_postfix(test_loss=round(running_loss / total, 5))

                if idx % 100 == 0:
                    wandb.log({'test_loss': running_loss / total})
        
