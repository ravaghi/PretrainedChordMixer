import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from utils.utils import init_run, init_weights, get_class_weights


@hydra.main(config_path="configs", version_base=None)
def main(config: DictConfig) -> None:
    device = init_run(config)

    # Instantiate model based on config
    model = instantiate(config=config.model).to(device)

    # Weight initialization
    if config.general.init_weights:
        model.apply(init_weights)

    # Loss function
    if config.general.compute_class_weights and "Plant" not in config.dataset.name:
        class_weights = get_class_weights(config.dataset.path, config.dataset.train_data)
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
        criterion = instantiate(config=config.loss, weight=class_weights)
    else:
        criterion = instantiate(config=config.loss)

    # Optimizer
    if config.general.name == "KeGRU":
        model_params = [model.hidden_weights, model.hidden_bias] + [param for param in model.parameters()]
    else:
        model_params = model.parameters()
    optimizer = instantiate(config=config.optimizer, params=model_params)

    # Dataloaders
    train_dataloader = instantiate(
        config=config.dataloader,
        dataset_filename=config.dataset.train_data,
        dataset_type=config.dataset.type,
        dataset_name=config.dataset.name
    ).create_dataloader()

    val_dataloader = instantiate(
        config=config.dataloader,
        dataset_filename=config.dataset.val_data,
        dataset_type=config.dataset.type,
        dataset_name=config.dataset.name
    ).create_dataloader()

    test_dataloader = instantiate(
        config=config.dataloader,
        dataset_filename=config.dataset.test_data,
        dataset_type=config.dataset.type,
        dataset_name=config.dataset.name
    ).create_dataloader()

    # Model trainer
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

    # Training and validation
    for epoch in range(1, config.general.max_epochs + 1):
        trainer.train(current_epoch_nr=epoch)
        trainer.evaluate(current_epoch_nr=epoch)

    # Testing
    trainer.test()


if __name__ == '__main__':
    main()
