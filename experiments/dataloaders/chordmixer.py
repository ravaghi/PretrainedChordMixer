from torch.utils.data import Dataset, DataLoader
import pandas as pd
import random
import torch
import os

from dataloader import Dataloader


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


class VEPDatasetCreator(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.reference = dataframe["reference"].values
        self.alternate = dataframe["alternate"].values
        self.tissue = dataframe["tissue"].values
        self.label = dataframe["label"].values

    def __getitem__(self, index):
        return self.reference[index], self.alternate[index], self.tissue[index], self.label[index]

    def __len__(self):
        return len(self.dataframe)


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
        X = torch.from_numpy(X)
        return X, Y, length, bin

    def __len__(self):
        return len(self.df)


class PlantDeepSeaDatasetCreator(Dataset):
    def __init__(self, df, batch_size, var_len=False):
        if var_len:
            target_list = df.columns.tolist()[:-3]
            df = complete_batch(df=df, batch_size=batch_size)
            columns = ['sequence', 'len', 'bin'] + target_list
            self.df = shuffle_batches(df=df)[columns]
            self.targets = self.df[target_list].values
        else:
            self.df = df

    def __getitem__(self, index):
        X = self.df.iloc[index]['sequence']
        length = self.df.iloc[index]['len']
        bin = self.df.iloc[index]['bin']
        Y = torch.FloatTensor(self.targets[index])
        X = torch.from_numpy(X)
        return X, Y, length, bin

    def __len__(self):
        return len(self.df)


class ChordMixerDataLoader(Dataloader):
    def create_dataloader(self):
        data_path = os.path.join(self.data_path, self.dataset)
        dataframe = pd.read_csv(data_path)

        if "Taxonomy" in self.dataset_name:
            dataframe = self.process_taxonomy_classification_dataframe(dataframe)
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

        if "Plant" in self.dataset_name:
            dataframe = self.process_plantdeepsea_dataframe(dataframe)
            dataset = PlantDeepSeaDatasetCreator(
                df=dataframe,
                batch_size=self.batch_size,
                var_len=True
            )
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False
            )

        if "Variant" in self.dataset_name:
            dataframe = self.process_variant_effect_prediction_dataframe(dataframe)
            dataset = VEPDatasetCreator(dataframe)
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=False
            )
