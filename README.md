# PretrainedChordMixer

![python](https://user-images.githubusercontent.com/44374191/224485239-3e013eff-f76e-46a4-90a7-e55fa0a6b3a7.svg)
![torch](https://user-images.githubusercontent.com/44374191/224485304-0b0f25c6-e31f-48a9-8cec-1767bffee1e6.svg)

---

- [Results](#results)
  - [Variant Effect Prediction in Human Genome](#variant-effect-prediction-in-human-genome)
  - [Open Chromatin Region Prediction in Plant Tissues](#open-chromatin-region-prediction-in-plant-tissues)
  - [DNA Sequence-Based Taxonomy Classification](#dna-sequence-based-taxonomy-classification)
- [Getting Started](#getting-started)
  - [Fine-tuning and Probing](#fine-tuning-and-probing)
  - [Pretraining](#pretraining)

---

## Results

### Variant Effect Prediction in Human Genome

| Dataset                   |   FineTunedChordMixer |   ProbedChordMixer |   ChordMixer |   KeGRU |DeeperDeepSEA |   Transformer |   Nyströmformer |   Poolformer |   Linformer |
|---------------------------|-----------------------|--------------------|--------------|---------|--------------|---------------|-----------------|--------------|-------------| 
| GRCh38                    |                 89.87 |              86.28 |        84.90 |   70.16 |        86.93 |         68.69 |           82.58 |        76.00 |       83.22 |



### Open Chromatin Region Prediction in Plant Tissues

| Dataset                   |   FineTunedChordMixer |   ProbedChordMixer |   ChordMixer |   KeGRU |DeeperDeepSEA |   Transformer |   Nyströmformer |   Poolformer |   Linformer |
|---------------------------|-----------------------|--------------------|--------------|---------|--------------|---------------|-----------------|--------------|-------------|     
| Arabidopsis Thaliana      |                 93.01 |              91.51 |        89.53 |   90.83 |        90.12 |         62.51 |           73.54 |        74.81 |       60.33 |
| Brachypodium Distachyon   |                 93.50 |              93.25 |        91.14 |   92.35 |        90.05 |         75.98 |           81.21 |        79.64 |       72.75 |
| Oryza Sativa MH           |                 93.99 |              92.73 |        90.95 |   92.15 |        91.02 |         71.40 |           77.89 |        74.56 |       62.39 |
| Oryza Sativa ZS           |                 93.52 |              92.40 |        90.79 |   92.14 |        89.70 |         75.01 |           76.85 |        76.20 |       71.04 |
| Setaria Italica           |                 94.50 |              93.18 |        91.93 |   92.99 |        91.92 |         82.31 |           83.55 |        81.22 |       70.98 |
| Sorghum Bicolor           |                 96.59 |              95.53 |        94.52 |   95.84 |        94.54 |         82.47 |           87.17 |        83.69 |       78.05 |
| Zea Mays                  |                 96.69 |              96.98 |        92.98 |   94.64 |        94.84 |         60.96 |           76.95 |        74.21 |       53.14 |



### DNA Sequence-Based Taxonomy Classification
| Dataset                   |   FineTunedChordMixer |   ProbedChordMixer |   ChordMixer |   KeGRU |DeeperDeepSEA |   Transformer |   Nyströmformer |   Poolformer |   Linformer |
|---------------------------|-----------------------|--------------------|--------------|---------|--------------|---------------|-----------------|--------------|-------------|
| Carassius vs. Labeo       |                 97.35 |              97.55 |        97.19 |   97.02 |        97.49 |         92.19 |           89.85 |        90.72 |       86.44 |
| Sus vs. Bos               |                 96.59 |              96.67 |        95.70 |   94.36 |        96.53 |         86.70 |           91.62 |        86.55 |       87.88 |
| Danio vs. Cyprinus        |                 98.67 |              98.62 |        98.93 |   98.74 |        99.12 |         90.06 |           94.31 |        92.81 |       87.00 |





## Getting started
All of the requirements and their versions are available in `requirements.txt`, and can be installed by running:
```bash
pip install -r requirements.txt
```

The configurations and hyperparameters can be found in the `configs` folder. As these are managed by [Hydra](https://hydra.cc/), they can simply be modified and overwritten, either directly in the config files, or by passing them as arguments to the training script, as follows:
```bash
python train.py --config-name=MAIN_CONFIG_NAME dataset=DATASET_CONFIG_NAME parameter=new_value
```
Note that `--config-name` and `dataset` always have to be passed to `train.py` as arguments. For a list of available config names and datasets, as well as other configuration parameters, run:
```bash
python train.py --help
```

### Fine-Tuning and Probing
Three pretrained models are availble for fine-tuning and probing. These can be found under `models` directory.
- `pcm-cl-1000-human.pt` trained on human reference genome GRCh38 
- `pcm-cl-1000-plant.pt` trained on plant DNA
- `pcm-vl.pt` trained on a dataset containing DNA sequences of varying lengths


Pretrained models can be instantiated and fine-tuned as follows: 
```python
class FineTunedChordMixer(nn.Module):
    def __init__(self, model_path, freeze, variable_length, n_class):
        super(FineTunedChordMixer, self).__init__()
        self.encoder = ChordMixerEncoder.from_pretrained(
            model_path=model_path,
            freeze=freeze,
            variable_length=variable_length
        )
        self.classifier = ...

    def forward(self, batch):
        ...
```
The model expectes one hot encoded DNA sequences as input. Run fine-tuning:
```
python train.py --config-name=chordmixer_finetuning dataset=DATASET_CONFIG_NAME
```

## Pretraining
Pretraining can be initiated using the following command:
```
python train.py --config-name=CONFIG_NAME
```
