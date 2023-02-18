from torch.utils.data import DataLoader
import pandas as pd

from .dataloader.dataloader import Dataloader
from .preprocessor.preprocessor import Preprocessor
from .chordmixer import (
    concater_collate,
    TaxonomyClassificationDataset,
    HumanVariantEffectPredictionDataset,
    PlantVariantEffectPredictionDataset
)


class FineTunedChordMixerDataLoader(Dataloader, Preprocessor):
    """ChordMixer finetuning dataloader class"""

    def create_taxonomy_classification_dataloader(self, dataframe: pd.DataFrame) -> DataLoader:
        dataframe = self.process_taxonomy_classification_dataframe(dataframe, model_name="FineTunedChordMixer")
        dataset = TaxonomyClassificationDataset(dataframe=dataframe, batch_size=self.batch_size)

        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=concater_collate
        )

    def create_human_variant_effect_prediction_dataloader(self, dataframe: pd.DataFrame) -> DataLoader:
        dataframe = self.process_human_variant_effect_prediction_dataframe(dataframe, model_name="FineTunedChordMixer")
        dataset = HumanVariantEffectPredictionDataset(dataframe=dataframe)

        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

    def create_plant_variant_effect_prediction_dataloader(self, dataframe: pd.DataFrame) -> DataLoader:
        dataframe = self.process_plant_variant_effect_prediction_dataframe(dataframe, model_name="FineTunedChordMixer")
        dataset = PlantVariantEffectPredictionDataset(dataframe=dataframe)

        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
