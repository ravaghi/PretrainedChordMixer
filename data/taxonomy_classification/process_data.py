import pandas as pd
import os

from experiments.dataloaders.dataloader import DNA_BASE_DICT_REVERSED


def pickle_to_csv(train: str, val: str, dataset_name: str) -> None:
    """
    Convert the pickled dataframes to csv files

    Args:
        train: path to the train dataframe
        val: path to the validation dataframe
        dataset_name: name of the dataset

    Returns:
        None
    """
    train_data = pd.read_pickle(train)
    val_data = pd.read_pickle(val)

    data = pd.concat([train_data, val_data])

    data = data.sample(frac=1).reset_index(drop=True)

    data["sequence"] = data["sequence"].apply(lambda x: "".join([DNA_BASE_DICT_REVERSED[i] for i in x]))
    data["bin"].fillna(0, inplace=True)

    train = data[:int(0.8 * len(data))]
    val = data[int(0.8 * len(data)):int(0.9 * len(data))]
    test = data[int(0.9 * len(data)):]

    os.mkdir(dataset_name)
    train.to_csv(f"{dataset_name}/{dataset_name}_train.csv", index=False)
    val.to_csv(f"{dataset_name}/{dataset_name}_val.csv", index=False)
    test.to_csv(f"{dataset_name}/{dataset_name}_test.csv", index=False)


pickle_to_csv("carassius_labeo_train.pkl", "carassius_labeo_val.pkl", "carassius_labeo")
pickle_to_csv("danio_cyprinus_train.pkl", "danio_cyprinus_val.pkl", "danio_cyprinus")
pickle_to_csv("sus_bos_train.pkl", "sus_bos_val.pkl", "sus_bos")
