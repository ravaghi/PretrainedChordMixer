from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import random
import torch
import os

DNA_BASE_DICT = {
    'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4, 'Y': 5, 'R': 6, 'M': 7,
    'W': 8, 'K': 9, 'S': 10, 'B': 11, 'H': 12, 'D': 13, 'V': 14
}
DNA_BASE_DICT_REVERSED = {
    0: 'A', 1: 'C', 2: 'G', 3: 'T', 4: 'N', 5: 'Y', 6: 'R', 7: 'M',
    8: 'W', 9: 'K', 10: 'S', 11: 'B', 12: 'H', 13: 'D', 14: 'V'
}

def complete_batch(df, batch_size):
    complete_bins = []
    bins = [bin_df for _, bin_df in df.groupby('bin')]

    for gr_id, bin in enumerate(bins):
        l = len(bin)
        remainder = l % batch_size
        integer = l // batch_size

        if remainder != 0:
            bin = pd.concat([bin, pd.concat([bin.iloc[:1]] * (batch_size - remainder))], ignore_index=True)
            integer += 1
        batch_ids = []
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


def process_plant_deepsea_dataframe(dataframe):
    dataframe["len"] = dataframe["sequence"].apply(len)
    dataframe["bin"] = -1
    dataframe["new_seq"] = dataframe["sequence"].apply(lambda x: np.array([DNA_BASE_DICT[i] for i in x]))
    dataframe.drop(columns=["sequence"], inplace=True)
    dataframe.rename(columns={"new_seq": "sequence"}, inplace=True)
    return dataframe[['sequence', 'len', 'bin', 'ATAC_7days_leaf_rep1', 'ATAC_7days_leaf_rep2',
       'ATAC_mesophyll_cell_rep1', 'ATAC_mesophyll_cell_rep2',
       'ATAC_mesophyll_cell_rep3', 'ATAC_root_hair_rep1',
       'ATAC_root_hair_rep2', 'ATAC_root_non_hair_rep1',
       'ATAC_root_non_hair_rep2', 'ATAC_root_tip_rep1', 'ATAC_root_tip_rep2',
       'ATAC_stem_cell_rep1', 'ATAC_stem_cell_rep2', 'ATAC_stem_cell_rep3',
       'DNase_flower_14_days', 'DNase_inflorescence_normal',
       'DNase_open_flower_normal', 'DNase_root_7_days',
       'DNase_seedling_normal']]


class DatasetCreator(Dataset):
    def __init__(self, df, batch_size, var_len=False):
        if var_len:
            target_list = ['ATAC_7days_leaf_rep1', 'ATAC_7days_leaf_rep2',
            'ATAC_mesophyll_cell_rep1', 'ATAC_mesophyll_cell_rep2',
            'ATAC_mesophyll_cell_rep3', 'ATAC_root_hair_rep1',
            'ATAC_root_hair_rep2', 'ATAC_root_non_hair_rep1',
            'ATAC_root_non_hair_rep2', 'ATAC_root_tip_rep1', 'ATAC_root_tip_rep2',
            'ATAC_stem_cell_rep1', 'ATAC_stem_cell_rep2', 'ATAC_stem_cell_rep3',
            'DNase_flower_14_days', 'DNase_inflorescence_normal',
            'DNase_open_flower_normal', 'DNase_root_7_days',
            'DNase_seedling_normal']
            df = complete_batch(df=df, batch_size=batch_size)
            self.df = shuffle_batches(df=df)[['sequence', 'len', 'bin', 'ATAC_7days_leaf_rep1', 'ATAC_7days_leaf_rep2',
                'ATAC_mesophyll_cell_rep1', 'ATAC_mesophyll_cell_rep2',
                'ATAC_mesophyll_cell_rep3', 'ATAC_root_hair_rep1',
                'ATAC_root_hair_rep2', 'ATAC_root_non_hair_rep1',
                'ATAC_root_non_hair_rep2', 'ATAC_root_tip_rep1', 'ATAC_root_tip_rep2',
                'ATAC_stem_cell_rep1', 'ATAC_stem_cell_rep2', 'ATAC_stem_cell_rep3',
                'DNase_flower_14_days', 'DNase_inflorescence_normal',
                'DNase_open_flower_normal', 'DNase_root_7_days',
                'DNase_seedling_normal']]
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


class ChordMixerDataLoader:
    def __init__(self, data_path, dataset, dataset_name, batch_size):
        self.data_path = data_path
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.batch_size = batch_size

    def create_dataloader(self):
        data_path = os.path.join(self.data_path, self.dataset)
        dataframe = pd.read_csv(data_path)[:100]

        if "Plant" in self.dataset_name:
            dataframe = process_plant_deepsea_dataframe(dataframe)        

        dataset = DatasetCreator(
            df=dataframe,
            batch_size=self.batch_size,
            var_len=True
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            #collate_fn=concater_collate,
            drop_last=False
        )
