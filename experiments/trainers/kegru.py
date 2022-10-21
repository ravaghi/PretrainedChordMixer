from tqdm import tqdm
from sklearn import metrics
import torch
import wandb
import numpy as np


class KeGruTrainer:
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
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        train_aucs = []

        loop = tqdm(enumerate(self.train_dataloader), total=num_batches)
        for idx, (x, y) in loop:
            x = x.to(self.device)
            y = y.to(self.device)

            y_hat = self.model(x)
                
            loss = torch.nn.functional.binary_cross_entropy(y_hat, y)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            total += y.size(0)
           
            predictions = y_hat.cpu().detach().numpy().reshape(y_hat.shape[0])
            targets = y.cpu().numpy().reshape(y_hat.shape[0])
            
            cur_auc = metrics.roc_auc_score(targets, predictions)
            train_aucs.append(cur_auc)
            
            cur_accuracy = (predictions > 0.5) == targets
            correct += np.sum(cur_accuracy)

            loop.set_description(f'Epoch {current_epoch_nr + 1}')
            loop.set_postfix(auc=round(cur_auc, 2))
            
            # if (idx + 1) % self.log_every_n_steps == 0:
            #     wandb.log({'train_loss': running_loss / (idx + 1)})
            #     wandb.log({'train_accuracy': correct / total})

        train_accuracy = correct / total
        train_loss = running_loss / num_batches
        train_auc = np.mean(train_aucs)
        wandb.log({'train_loss': train_loss}, step=current_epoch_nr)
        wandb.log({'train_accuracy': train_accuracy}, step=current_epoch_nr)
        wandb.log({'train_auc': train_auc}, step=current_epoch_nr)

    def evaluate(self, current_epoch_nr):
        self.model.eval()

        num_batches = len(self.val_dataloader)

        running_loss = 0
        correct = 0
        total = 0
        
        train_aucs = []

        with torch.no_grad():
            loop = tqdm(enumerate(self.val_dataloader), total=num_batches)
            for idx, (x, y,) in loop:
                x = x.to(self.device)
                y = y.to(self.device)

                y_hat = self.model(x)
                
                loss = torch.nn.functional.binary_cross_entropy(y_hat, y)
                
                running_loss += loss.item()
                total += y.size(0)
            
                predictions = y_hat.cpu().detach().numpy().reshape(y_hat.shape[0])
                targets = y.cpu().numpy().reshape(y_hat.shape[0])
                
                cur_auc = metrics.roc_auc_score(targets, predictions)
                train_aucs.append(cur_auc)
                
                cur_accuracy = (predictions > 0.5) == targets
                correct += np.sum(cur_accuracy)

                loop.set_description(f'Epoch {current_epoch_nr + 1}')
                loop.set_postfix(auc=round(cur_auc, 2))
                
                # if (idx + 1) % self.log_every_n_steps == 0:
                #     wandb.log({'val_loss': running_loss / (idx + 1)})
                #     wandb.log({'val_accuracy': correct / total})

        validation_accuracy = correct / total
        validation_loss = running_loss / num_batches
        val_auc = np.mean(train_aucs)
        wandb.log({'val_loss': validation_loss}, step=current_epoch_nr)
        wandb.log({'val_accuracy': validation_accuracy}, step=current_epoch_nr)
        wandb.log({'val_auc': val_auc}, step=current_epoch_nr)
