from collections import OrderedDict
import torch
from model import ChordMixerEncoder
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import random
from tqdm import tqdm
import os
from sklearn import metrics
import wandb

def load_encoder():
    # Load pytorch saved model from models/
    model = torch.load('models/2023-01-31_0513-PretrainedChordMixer-AUC-None.pth')

    state_dict = OrderedDict()
    for key, value in model.items():
        if "encoder" in key:
            new_key = key.replace("module.encoder.", "")
            state_dict[new_key] = value


    chordmixer = ChordMixerEncoder(4, 40_000, 16, 196, 450, 0, 0, True)
    chordmixer.load_state_dict(state_dict)

    return chordmixer


class ChordMixerNet(torch.nn.Module):
    def __init__(self):
        super(ChordMixerNet, self).__init__()
        self.encoder = load_encoder()
        self.classifier = torch.nn.Linear(450, 2)

        # for param in self.encoder.parameters():
        #     param.requires_grad = False

    def forward(self, x, lengths=None):
        encoded = self.encoder(x, lengths)
        return self.classifier(encoded)

def complete_batch(df, batch_size):
    complete_bins = []
    bins = [bin_df for _, bin_df in df.groupby('bin')]

    for gr_id, bin in enumerate(bins):
        l = len(bin)
        remainder = l % batch_size
        integer = l // batch_size

        if remainder != 0:
            # take the first example and copy (batch_size - remainder) times
            bin = pd.concat([bin, pd.concat([bin.iloc[:1]] * (batch_size - remainder))], ignore_index=True)
            integer += 1
        batch_ids = []
        # create indices
        for i in range(integer):
            batch_ids.extend([f'{i}_bin{gr_id}'] * batch_size)
        bin['batch_id'] = batch_ids
        complete_bins.append(bin)
    return pd.concat(complete_bins, ignore_index=True)


def shuffle_batches(df):
    batch_bins = [df_new for _, df_new in df.groupby('batch_id')]
    random.shuffle(batch_bins)
    return pd.concat(batch_bins).reset_index(drop=True)


def concater_collate(batch):
    (xx, yy, lengths, bins) = zip(*batch)
    xx = torch.cat(xx, 0)
    yy = torch.tensor(yy)
    return xx, yy, list(lengths), list(bins)


class TaxonomyDatasetCreator(Dataset):
    def __init__(self, df, batch_size, var_len=False):
        if var_len:
            df = complete_batch(df=df, batch_size=batch_size)
            self.df = shuffle_batches(df=df)[['sequence', 'label', 'len', 'bin']]
        else:
            self.df = df

    def __getitem__(self, index):
        X, Y, length, bin = self.df.iloc[index, :]
        Y = torch.tensor(Y)
        X = torch.from_numpy(X).float()
        return X, Y, length, bin

    def __len__(self):
        return len(self.df)


class ChordMixerDataLoader:
    def __init__(self, dataset_filename):
        self.data_path = "data/taxonomy_classification/carassius_labeo"
        self.dataset_filename = dataset_filename
        self.batch_size = 2

    def create_dataloader(self):
        data_path = os.path.join(self.data_path, self.dataset_filename)
        dataframe = pd.read_csv(data_path)

        def _filter_bases(sequence):
            bases = ["A", "C", "G", "T"]
            base_dict = {"A": 0, "C": 1, "G": 2, "T": 3}
            base_index_list = torch.tensor([base_dict[base] for base in sequence if base in bases], dtype=torch.int64)
            return torch.nn.functional.one_hot(base_index_list, num_classes=4).numpy()
        dataframe["sequence"] = dataframe["sequence"].apply(_filter_bases)
        dataframe["len"] = dataframe["sequence"].apply(len)

        dataset = TaxonomyDatasetCreator(
            df=dataframe,
            batch_size=self.batch_size,
            var_len=True
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=concater_collate,
            drop_last=False
        )


def train(epoch):
    model.train()

    num_batches = len(train_dataloader)

    running_loss = 0.0
    correct = 0
    total = 0

    preds = []
    targets = []

    loop = tqdm(train_dataloader, total=num_batches)
    for batch in loop:
        x, y, seq_len, bin = batch
        x = x.to(device)
        y = y.to(device)

        y_hat = model(x, seq_len)

        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()

        _, predicted = y_hat.max(1)
        correct_predictions = predicted.eq(y).sum().item()

        correct += correct_predictions
        total += y.size(0)

        targets.extend(y.detach().cpu().numpy())
        preds.extend(predicted.detach().cpu().numpy())

        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(train_acc=round(correct / total, 3),
                            train_loss=round(running_loss / total, 3))

    train_auc = metrics.roc_auc_score(targets, preds)
    train_accuracy = correct / total
    train_loss = running_loss / num_batches

    wandb.log({'train_auc': train_auc}, step=epoch)
    wandb.log({'train_accuracy': train_accuracy}, step=epoch)
    wandb.log({'train_loss': train_loss}, step=epoch)

def validate(epoch):
    model.eval()

    num_batches = len(val_dataloader)

    running_loss = 0.0
    correct = 0
    total = 0

    preds = []
    targets = []

    with torch.no_grad():
        loop = tqdm(val_dataloader, total=num_batches)
        for batch in loop:
            x, y, seq_len, bin = batch
            x = x.to(device)
            y = y.to(device)

            y_hat = model(x, seq_len)

            loss = criterion(y_hat, y)

            running_loss += loss.item()

            _, predicted = y_hat.max(1)
            correct_predictions = predicted.eq(y).sum().item()

            correct += correct_predictions
            total += y.size(0)

            targets.extend(y.detach().cpu().numpy())
            preds.extend(predicted.detach().cpu().numpy())

            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(val_acc=round(correct / total, 3),
                                val_loss=round(running_loss / total, 3))

    val_auc = metrics.roc_auc_score(targets, preds)
    val_accuracy = correct / total
    val_loss = running_loss / num_batches

    wandb.log({'val_auc': val_auc}, step=epoch)
    wandb.log({'val_accuracy': val_accuracy}, step=epoch)
    wandb.log({'val_loss': val_loss}, step=epoch)

def test():
    model.eval()

    num_batches = len(test_dataloader)

    running_loss = 0.0
    correct = 0
    total = 0

    preds = []
    targets = []

    with torch.no_grad():
        loop = tqdm(test_dataloader, total=num_batches)
        for batch in loop:
            x, y, seq_len, bin = batch
            x = x.to(device)
            y = y.to(device)

            y_hat = model(x, seq_len)

            loss = criterion(y_hat, y)

            running_loss += loss.item()

            _, predicted = y_hat.max(1)
            correct_predictions = predicted.eq(y).sum().item()

            correct += correct_predictions
            total += y.size(0)

            targets.extend(y.detach().cpu().numpy())
            preds.extend(predicted.detach().cpu().numpy())

            loop.set_description(f'Testing')
            loop.set_postfix(test_acc=round(correct / total, 3),
                                test_loss=round(running_loss / total, 3))

    test_auc = metrics.roc_auc_score(targets, preds)
    test_accuracy = correct / total
    test_loss = running_loss / num_batches

    wandb.run.summary["test_auc"] = test_auc
    wandb.run.summary["test_accuracy"] = test_accuracy
    wandb.run.summary["test_loss"] = test_loss


if "__main__" == __name__:
    device = "cuda:0"

    model = ChordMixerNet().to(device)
    
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    optimizer = torch.optim.Adam(lr=0.0001, params=model.parameters())

    train_dataloader = ChordMixerDataLoader("carassius_labeo_train.csv").create_dataloader()
    val_dataloader = ChordMixerDataLoader("carassius_labeo_val.csv").create_dataloader()
    test_dataloader = ChordMixerDataLoader("carassius_labeo_test.csv").create_dataloader()  

    wandb.init(project="PDT", entity="ravaghi", name="PretrainedChordMixer-TaxonomyClassification-CarassiusVsLabeo")

    for epoch in range(1, 31):
        train(epoch)
        validate(epoch)
    test()

