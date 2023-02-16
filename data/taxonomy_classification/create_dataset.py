import pandas as pd
import os

DNA_BASE_DICT_REVERSED = {
    0: 'A', 1: 'C', 2: 'G', 3: 'T', 4: 'N', 5: 'Y', 6: 'R', 7: 'M',
    8: 'W', 9: 'K', 10: 'S', 11: 'B', 12: 'H', 13: 'D', 14: 'V'
}


def pickle_to_parquet(train_data: str, test_data: str, dataset_name: str) -> None:
    """
    Converts and joins two pickled dataframes to parquet format and splits them into train, val and test sets.

    Args:
        train_data: path to the train dataframe
        test_data: path to the test dataframe
        dataset_name: name of the new dataset

    Returns:
        None
    """
    print(f"Processing {dataset_name} datasets...")

    train_data = pd.read_pickle(train_data)
    test_data = pd.read_pickle(test_data)

    dataframe = pd.concat([train_data, test_data])

    dataframe = dataframe.sample(frac=1).reset_index(drop=True)

    dataframe["sequence"] = dataframe["sequence"].apply(
        lambda x: "".join([DNA_BASE_DICT_REVERSED[i] for i in x])).astype(str)
    dataframe["label"] = dataframe["label"].astype("int8")

    dataframe = dataframe.drop(columns=["bin", "len"])

    train = dataframe[:int(0.8 * len(dataframe))]
    val = dataframe[int(0.8 * len(dataframe)):int(0.9 * len(dataframe))]
    test = dataframe[int(0.9 * len(dataframe)):]

    if not os.path.exists(dataset_name):
        os.mkdir(dataset_name)

    train.to_parquet(f"{dataset_name}/{dataset_name}_train.parquet", index=False)
    val.to_parquet(f"{dataset_name}/{dataset_name}_val.parquet", index=False)
    test.to_parquet(f"{dataset_name}/{dataset_name}_test.parquet", index=False)


if __name__ == "__main__":
    pickle_to_parquet("carassius_labeo_train.pkl", "carassius_labeo_test.pkl", "carassius_labeo")
    pickle_to_parquet("danio_cyprinus_train.pkl", "danio_cyprinus_test.pkl", "danio_cyprinus")
    pickle_to_parquet("sus_bos_train.pkl", "sus_bos_test.pkl", "sus_bos")
