# PretrainedChordMixer

![python](https://user-images.githubusercontent.com/44374191/224485239-3e013eff-f76e-46a4-90a7-e55fa0a6b3a7.svg)
![torch](https://user-images.githubusercontent.com/44374191/224485304-0b0f25c6-e31f-48a9-8cec-1767bffee1e6.svg)



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

## Fine-tuning
The pretrained model can be instantiated and fine-tuned as follows: 
```python
class FineTunedChordMixer(nn.Module):
    def __init__(self, model_path, freeze, variable_length, n_class):
        super(FineTunedChordMixer, self).__init__()
        self.encoder = ChordMixerEncoder.from_pretrained(
            model_path=model_path,
            freeze=freeze,
            variable_length=variable_length
        )
        self.classifier = nn.Linear(self.encoder.prelinear_out_features, n_class)

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
python train.py --config-name=chordmixer_pretraining
```
Note that `dataset` doesn't have to be passed to the script, as there is only one available option for pretraining.
