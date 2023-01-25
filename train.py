import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import torch
from datetime import datetime

from experiments.utils.utils import init_run


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(config: DictConfig) -> None:
    device = init_run(config)

    dataloader = instantiate(config=config.dataloader)
    train_dataloader, val_dataloader, test_dataloader = dataloader.create_dataloaders()

    model = torch.nn.DataParallel(instantiate(config=config.model)).to(device)
    criterion = instantiate(config=config.loss)
    optimizer = instantiate(config=config.optimizer, params=model.parameters())

    trainer = instantiate(
        config=config.trainer,
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        device=device,
        criterion=criterion,
        optimizer=optimizer
    )

    for epoch in range(1, config.general.max_epochs + 1):
        trainer.train(current_epoch_nr=epoch)
        trainer.evaluate(current_epoch_nr=epoch)
    trainer.test()
    
    torch.save(model.state_dict(), f"models/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pth")


if __name__ == '__main__':
    main()
