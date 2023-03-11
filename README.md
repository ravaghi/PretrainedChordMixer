# PretrainedChordMixer
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="100.0" height="20"><linearGradient id="smooth" x2="0" y2="100%"><stop offset="0" stop-color="#bbb" stop-opacity=".1"/><stop offset="1" stop-opacity=".1"/></linearGradient><clipPath id="round"><rect width="100.0" height="20" rx="3" fill="#fff"/></clipPath><g clip-path="url(#round)"><rect width="65.5" height="20" fill="#555"/><rect x="65.5" width="34.5" height="20" fill="#007ec6"/><rect width="100.0" height="20" fill="url(#smooth)"/></g><g fill="#fff" text-anchor="middle" font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="110"><image x="5" y="3" width="14" height="14" xlink:href="https://dev.w3.org/SVG/tools/svgweb/samples/svg-files/python.svg"/><text x="422.5" y="150" fill="#010101" fill-opacity=".3" transform="scale(0.1)" textLength="385.0" lengthAdjust="spacing">python</text><text x="422.5" y="140" transform="scale(0.1)" textLength="385.0" lengthAdjust="spacing">python</text><text x="817.5" y="150" fill="#010101" fill-opacity=".3" transform="scale(0.1)" textLength="245.0" lengthAdjust="spacing">3.10</text><text x="817.5" y="140" transform="scale(0.1)" textLength="245.0" lengthAdjust="spacing">3.10</text><a xlink:href=""><rect width="65.5" height="20" fill="rgba(0,0,0,0)"/></a><a xlink:href="https://www.python.org/"><rect x="65.5" width="34.5" height="20" fill="rgba(0,0,0,0)"/></a></g></svg>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="100.5" height="20"><linearGradient id="smooth" x2="0" y2="100%"><stop offset="0" stop-color="#bbb" stop-opacity=".1"/><stop offset="1" stop-opacity=".1"/></linearGradient><clipPath id="round"><rect width="100.5" height="20" rx="3" fill="#fff"/></clipPath><g clip-path="url(#round)"><rect width="55.5" height="20" fill="#555"/><rect x="55.5" width="45.0" height="20" fill="#007ec6"/><rect width="100.5" height="20" fill="url(#smooth)"/></g><g fill="#fff" text-anchor="middle" font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="110"><image x="5" y="3" width="14" height="14" xlink:href="https://upload.wikimedia.org/wikipedia/commons/1/10/PyTorch_logo_icon.svg"/><text x="372.5" y="150" fill="#010101" fill-opacity=".3" transform="scale(0.1)" textLength="285.0" lengthAdjust="spacing">torch</text><text x="372.5" y="140" transform="scale(0.1)" textLength="285.0" lengthAdjust="spacing">torch</text><text x="770.0" y="150" fill="#010101" fill-opacity=".3" transform="scale(0.1)" textLength="350.0" lengthAdjust="spacing">1.13.1</text><text x="770.0" y="140" transform="scale(0.1)" textLength="350.0" lengthAdjust="spacing">1.13.1</text><a xlink:href=""><rect width="55.5" height="20" fill="rgba(0,0,0,0)"/></a><a xlink:href="https://pytorch.org/"><rect x="55.5" width="45.0" height="20" fill="rgba(0,0,0,0)"/></a></g></svg>

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