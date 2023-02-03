import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import torch
from datetime import datetime
import numpy as np

from experiments.utils.utils import init_run


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(config: DictConfig) -> None:
    device = init_run(config)

    model = torch.nn.DataParallel(instantiate(config=config.model)).to(device)
    criterion = instantiate(config=config.loss)
    optimizer = instantiate(config=config.optimizer, params=model.parameters())

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"Number of trainable parameters: {params:,}")

    dataloader = instantiate(config=config.dataloader)
    train_dataloader, test_dataloader = dataloader.create_dataloaders()

    trainer = instantiate(
        config=config.trainer,
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        device=device,
        criterion=criterion,
        optimizer=optimizer
    )

    for epoch in range(1, config.general.max_epochs + 1):
        trainer.train(current_epoch_nr=epoch)
    test_auc = trainer.test()

    model_name = f"{datetime.now().strftime('%Y-%m-%d_%H%M')}-PretrainedChordMixer-AUC-{test_auc}"
    torch.save(model.state_dict(), f"models/{model_name}.pth")


if __name__ == '__main__':
    main()
