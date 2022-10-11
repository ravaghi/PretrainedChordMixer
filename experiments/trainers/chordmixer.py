from tqdm import tqdm
import torch
import wandb


class ChordMixerTrainer:
    def __init__(self, model, train_dataloader, val_dataloader, device, criterion, optimizer):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer

    def train(self, current_epoch_nr):
        self.model.train()

        running_accuracy = 0.0
        running_loss = 0.0
        num_batches = len(self.train_dataloader)

        loop = tqdm(enumerate(self.train_dataloader), total=num_batches)
        for idx, (x, y, seq_len, _bin) in loop:
            x = x.to(self.device)
            y = y.to(self.device)

            y_hat = self.model(x, seq_len)
            loss = self.criterion(y_hat, y)

            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            _, predicted = y_hat.max(1)
            running_accuracy += (predicted == y).sum().item()
            running_loss += loss.item()

            loop.set_description(f'Epoch {current_epoch_nr + 1}')
            loop.set_postfix(train_acc=round(running_accuracy / (idx + 1), 2),
                             train_loss=round(running_loss / (idx + 1), 2))

        train_accuracy = running_accuracy / num_batches
        train_loss = running_loss / num_batches
        wandb.log({'train_loss': train_loss})
        wandb.log({'train_accuracy': train_accuracy})

    def evaluate(self, current_epoch_nr):
        self.model.eval()

        running_accuracy = 0.0
        running_loss = 0.0
        num_batches = len(self.val_dataloader)

        with torch.no_grad():
            loop = tqdm(enumerate(self.val_dataloader), total=num_batches, position=0, leave=True, ascii=False)
            for idx, (x, y, seq_len, _bin) in loop:
                x = x.to(self.device)
                y = y.to(self.device)

                y_hat = self.model(x, seq_len)
                loss = self.criterion(y_hat, y)

                _, predicted = y_hat.max(1)
                running_accuracy += (predicted == y).sum().item()
                running_loss += loss.item()

                loop.set_description(f'Epoch {current_epoch_nr + 1}')
                loop.set_postfix(val_acc=round(running_accuracy / (idx + 1), 2),
                                 val_loss=round(running_loss / (idx + 1), 2))

        validation_accuracy = running_accuracy / num_batches
        validation_loss = running_loss / num_batches
        wandb.log({'validation_loss': validation_loss})
        wandb.log({'validation_accuracy': validation_accuracy})
