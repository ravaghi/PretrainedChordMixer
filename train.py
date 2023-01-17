import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from experiments.utils.utils import init_run


@hydra.main(version_base=None, config_path="configs", config_name="pretrained_chordmixer")
def main(config: DictConfig) -> None:
    device = init_run(config)

    dataloader = instantiate(config=config.dataloader)
    train_dataloader, val_dataloader, test_dataloader = dataloader.create_dataloaders()

    print("Train dataloader:")
    train_sample = next(iter(train_dataloader))
    print(len(train_dataloader))
    print(train_sample[0].shape)
    print(train_sample[1].shape)
    print(train_sample[2].shape)

    print("Val dataloader:")
    val_sample = next(iter(val_dataloader))
    print(len(val_dataloader))
    print(val_sample[0].shape)
    print(val_sample[1].shape)
    print(val_sample[2].shape)

    print("Test dataloader:")
    test_sample = next(iter(test_dataloader))
    print(len(test_dataloader))
    print(test_sample[0].shape)
    print(test_sample[1].shape)
    print(test_sample[2].shape)

    exit()

    model = instantiate(config=config.model).to(device)
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
        optimizer=optimizer,
        task=config.dataset.type
    )

    for epoch in range(1, config.general.max_epochs + 1):
        trainer.train(current_epoch_nr=epoch)
        trainer.evaluate(current_epoch_nr=epoch)

    trainer.test()


if __name__ == '__main__':
    main()
