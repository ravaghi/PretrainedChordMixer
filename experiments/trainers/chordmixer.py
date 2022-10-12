from tqdm import tqdm
import torch
import wandb


class ChordMixerTrainer:
    def __init__(self, model, train_dataloader, val_dataloader, device, criterion, optimizer, log_every_n_steps):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.log_every_n_steps = log_every_n_steps
        
    def train(self, current_epoch_nr):
        self.model.train()
        
        num_batches = len(self.train_dataloader)
        
        running_loss = 0
        correct = 0
        total = 0

        loop = tqdm(enumerate(self.train_dataloader), total=num_batches)
        for idx, (x, y, seq_len, _bin) in loop:
            x = x.to(self.device)
            y = y.to(self.device)

            y_hat = self.model(x, seq_len)
            loss = self.criterion(y_hat, y)

            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()
            
            running_loss += loss.item()

            _, predicted = y_hat.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

            loop.set_description(f'Epoch {current_epoch_nr + 1}')
            loop.set_postfix(train_acc=round(correct / total, 2),
                             train_loss=round(running_loss / total, 2))
            
            if (idx + 1) % self.log_every_n_steps == 0:
                wandb.log({'train_loss': running_loss / (idx + 1)})
                wandb.log({'train_accuracy': correct / total})

        train_accuracy = correct / total
        train_loss = running_loss / num_batches
        wandb.log({'train_loss': train_loss})
        wandb.log({'train_accuracy': train_accuracy})

    def evaluate(self, current_epoch_nr):
        self.model.eval()

        num_batches = len(self.val_dataloader)

        running_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            loop = tqdm(enumerate(self.val_dataloader), total=num_batches, position=0, leave=True, ascii=False)
            for idx, (x, y, seq_len, _bin) in loop:
                x = x.to(self.device)
                y = y.to(self.device)

                y_hat = self.model(x, seq_len)
                loss = self.criterion(y_hat, y)
                
                running_loss += loss.item()

                _, predicted = y_hat.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()

                loop.set_description(f'Epoch {current_epoch_nr + 1}')
                loop.set_postfix(val_acc=round(correct / total, 2),
                                 val_loss=round(running_loss / total, 2))
                
                if (idx + 1) % self.log_every_n_steps == 0:
                    wandb.log({'val_loss': running_loss / (idx + 1)})
                    wandb.log({'val_accuracy': correct / total})

        validation_accuracy = correct / total
        validation_loss = running_loss / num_batches
        wandb.log({'validation_loss': validation_loss})
        wandb.log({'validation_accuracy': validation_accuracy})
