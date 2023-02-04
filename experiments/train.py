import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from utils.utils import init_run


@hydra.main(config_path="configs", version_base=None)
def main(config: DictConfig) -> None:
    device = init_run(config)

    model = instantiate(config=config.model).to(device)

    criterion = instantiate(config=config.loss)

    if config.general.name == "KeGRU":
        model_params = [model.hidden_weights, model.hidden_bias] + [param for param in model.parameters()]
        optimizer = instantiate(config=config.optimizer, params=model_params)
    else:
        optimizer = instantiate(config=config.optimizer, params=model.parameters())

    dataloader = instantiate(
        config=config.dataloader,
        dataset_type=config.dataset.type,
        dataset_name=config.dataset.name,
        train_dataset=config.dataset.train_data,
        val_dataset=config.dataset.val_data,
        test_dataset=config.dataset.test_data
    )
    train_dataloader, val_dataloader, test_dataloader = dataloader.create_dataloaders()

    trainer = instantiate(
        config=config.trainer,
        device=device,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        task=config.dataset.type,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader
    )

    for epoch in range(1, config.general.max_epochs + 1):
        trainer.train(current_epoch_nr=epoch)
        trainer.evaluate(current_epoch_nr=epoch)

    trainer.test()


if __name__ == '__main__':
    main()
