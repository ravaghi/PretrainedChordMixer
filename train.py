import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.nn import DataParallel

from utils.utils import init_run, print_model_params


@hydra.main(config_path="configs", version_base=None)
def main(config: DictConfig) -> None:
    device = init_run(config)

    if config.general.use_multi_gpu:
        model = DataParallel(instantiate(config=config.model)).to(device)
    else:
        model = instantiate(config=config.model).to(device)
    
    print_model_params(model)

    criterion = instantiate(config=config.loss)
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
        test_dataloader=test_dataloader,
        log_to_wandb=config.general.log_to_wandb,
        save_dir=config.general.save_dir,
        scheduler=None
    )

    for epoch in range(1, config.general.max_epochs + 1):
        trainer.train(current_epoch_nr=epoch)
        trainer.evaluate(current_epoch_nr=epoch)

    trainer.test()


if __name__ == '__main__':
    main()
