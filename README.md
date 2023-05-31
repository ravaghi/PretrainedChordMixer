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

| **Model/Dataset**       | **GRCh38** |
|-------------------------|:----------:|
| **FineTunedChordMixer** |    89.87   |
| **ProbedChordMixer**    |    86.28   |
| **ChordMixer**          |    84.90   |
| **KeGRU**               |    70.16   |
| **DeeperDeepSEA**       |    86.93   |
| **Transformer**         |    68.69   |
| **Nyströmformer**       |    82.58   |
| **Poolformer**          |    76.00   |
| **Linformer**           |    83.22   |


### Open Chromatin Region Prediction in Plant Tissues

| **Model/Dataset**       | **A. Thaliana** | **B. Distachyon** | **O. Sativa MH** | **O.Sativa ZS** | **S. Italica** | **S. Bicolor** | **Z. Mays** |
|-------------------------|:---------------:|:-----------------:|:----------------:|:---------------:|:--------------:|:--------------:|:-----------:|
| **FineTunedChordMixer** |      93.01      |       93.50       |       93.99      |      93.52      |      94.50     |      96.59     |    96.69    |
| **ProbedChordMixer**    |      91.51      |       93.25       |       92.73      |      92.40      |      93.18     |      95.53     |    96.98    |
| **ChordMixer**          |      89.53      |       91.14       |       90.95      |      90.79      |      91.93     |      94.52     |    92.98    |
| **KeGRU**               |      90.83      |       92.35       |       92.15      |      92.14      |      92.99     |      95.84     |    94.64    |
| **DeeperDeepSEA**       |      90.12      |       90.05       |       91.02      |      89.70      |      91.92     |      94.54     |    94.84    |
| **Transformer**         |      62.51      |       75.98       |       71.40      |      75.01      |      82.31     |      82.47     |    60.96    |
| **Nyströmformer**       |      73.54      |       81.21       |       77.89      |      76.85      |      83.55     |      87.17     |    76.95    |
| **Poolformer**          |      74.81      |       79.64       |       74.56      |      76.20      |      81.22     |      83.69     |    74.21    |
| **Linformer**           |      60.33      |       72.75       |       62.39      |      71.04      |      70.98     |      78.05     |    53.14    |



### DNA Sequence-Based Taxonomy Classification
| **Model/Dataset**       | **Carassius vs. Labeo** | **Sus vs. Bos** | **Danio vs. Cyprinus** |
|-------------------------|:-----------------------:|:---------------:|:----------------------:|
| **FineTunedChordMixer** |          97.35          |      96.59      |          98.67         |
| **ProbedChordMixer**    |          97.55          |      96.67      |          98.62         |
| **ChordMixer**          |          97.19          |      95.70      |          98.93         |
| **KeGRU**               |          97.02          |      94.36      |          98.74         |
| **DeeperDeepSEA**       |          97.49          |      96.53      |          99.12         |
| **Transformer**         |          92.19          |      86.70      |          90.06         |
| **Nyströmformer**       |          89.85          |      91.62      |          94.31         |
| **Poolformer**          |          90.72          |      86.55      |          92.81         |
| **Linformer**           |          86.44          |      87.88      |          87.00         |





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
